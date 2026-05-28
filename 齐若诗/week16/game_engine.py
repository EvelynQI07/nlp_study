"""游戏引擎

负责驱动狼人杀游戏的完整流程：
1. 初始化：角色分配、玩家创建
2. 夜晚阶段：狼人击杀、预言家查验、女巫用药、猎人开枪
3. 白天阶段：公布死亡、公开发言、投票处决
4. 胜负判定：每个阶段后检查游戏是否结束

信息隔离：每个 Agent 只收到属于自己视角的信息。
"""
from __future__ import annotations

import asyncio
import random
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from roles import create_role, ROLE_CN_NAMES
from agent.player_agent import PlayerAgent, DECISION_STYLES
from agent.summary_agent import SummaryAgent
from schema.game_record import GameRecord, DialogueRecord, PlayerSummary
from schema.game_logger import GameLogger


# ── 游戏配置 ─────────────────────────────────────────────────────────────────

GAME_CONFIGS: Dict[str, Dict[str, Any]] = {
    "standard_6": {
        "name": "标准6人局",
        "roles": ["werewolf", "werewolf", "seer", "witch", "hunter", "villager"],
    },
    "simple_4": {
        "name": "简单4人局",
        "roles": ["werewolf", "seer", "witch", "villager"],
    },
    "big_9": {
        "name": "大型9人局",
        "roles": ["werewolf", "werewolf", "werewolf", "seer", "witch", "hunter",
                  "villager", "villager", "villager"],
    },
}


def get_role_config(config_name: str) -> List[str]:
    """获取角色配置列表"""
    cfg = GAME_CONFIGS.get(config_name)
    if not cfg:
        raise ValueError(f"未知配置: {config_name}，可用: {list(GAME_CONFIGS.keys())}")
    return cfg["roles"]


def shuffle_roles(roles: List[str]) -> Dict[int, str]:
    """打乱角色后返回 {player_id: role_type} 字典"""
    shuffled = roles[:]
    random.shuffle(shuffled)
    return {i: r for i, r in enumerate(shuffled)}


# ── 玩家状态 ─────────────────────────────────────────────────────────────────

class PlayerState:
    """单个玩家的运行时状态"""

    def __init__(self, player_id: int, name: str, role_type: str):
        self.player_id = player_id
        self.name = name
        self.role_type = role_type
        self.role = create_role(role_type)
        self.role_name = ROLE_CN_NAMES.get(role_type, role_type)
        self.camp = self.role.camp
        self.alive = True
        self.poisoned = False          # 被女巫毒死（不能触发猎人技能）
        self.agent: Optional[PlayerAgent] = None

        # 女巫专用状态
        self.has_heal = (role_type == "witch")
        self.has_poison = (role_type == "witch")
        # 预言家已查验记录 {player_id: "good"/"evil"}
        self.verified: Dict[int, str] = {}


# ── 游戏引擎 ─────────────────────────────────────────────────────────────────

class GameEngine:
    """
    狼人杀游戏引擎

    对外提供:
    - initialize(role_assignment, player_styles): 初始化游戏
    - start(): 运行完整游戏直到结束
    - step(): 执行下一个阶段（用于 API 逐步控制）
    """

    def __init__(
        self,
        game_id: str,
        player_names: Optional[List[str]] = None,
        logger: Optional[GameLogger] = None,
        on_dialogue: Optional[Callable[[DialogueRecord], None]] = None,
    ):
        self.game_id = game_id
        self.logger = logger or GameLogger(game_id)
        self.on_dialogue = on_dialogue  # 实时回调（用于 SSE 推送）

        self.players: Dict[int, PlayerState] = {}
        self.player_names = player_names or []
        self.day = 0
        self.phase = "init"
        self.running = False
        self.paused = False
        self.finished = False
        self.winner: Optional[str] = None

        self.dialogues: List[DialogueRecord] = []
        self.death_records: List[Dict[str, Any]] = []
        self.summaries: List[PlayerSummary] = []

        # 夜晚暂存
        self._night_kill_target: Optional[int] = None
        self._witch_heal: bool = False
        self._witch_poison_target: Optional[int] = None

        # 阶段事件（用于 API 暂停/继续）
        self._step_event: asyncio.Event = asyncio.Event()
        self._step_event.set()  # 默认不暂停

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def initialize(self, role_assignment: Dict[int, str], player_styles: Dict[int, str]) -> None:
        """初始化玩家和 Agent"""
        n = len(role_assignment)
        if not self.player_names or len(self.player_names) < n:
            self.player_names = [f"玩家{i+1}" for i in range(n)]

        for player_id, role_type in role_assignment.items():
            name = self.player_names[player_id]
            ps = PlayerState(player_id, name, role_type)
            self.players[player_id] = ps

        # 为每个玩家创建私有上下文和 Agent
        all_players_info = {
            pid: {"role_type": p.role_type, "name": p.name}
            for pid, p in self.players.items()
        }
        for player_id, ps in self.players.items():
            style = player_styles.get(player_id, "balanced")
            private_ctx = ps.role.get_private_context(player_id, all_players_info)
            # 预言家需要知道自己的已查验记录引用
            if ps.role_type == "seer":
                private_ctx["verified_players"] = ps.verified

            ps.agent = PlayerAgent(
                player_id=player_id,
                player_name=ps.name,
                role_name=ps.role_name,
                role_type=ps.role_type,
                camp=ps.camp,
                private_context=private_ctx,
                decision_style=style,
            )
            self.logger.info(f"[初始化] {ps.name}({ps.role_name}) 风格:{style}")

        self.phase = "ready"
        self.logger.info(f"游戏初始化完成，共 {n} 名玩家")

    # ── 主游戏循环 ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """运行完整游戏"""
        self.running = True
        self.logger.info("===== 游戏开始 =====")

        while not self.finished:
            await self._wait_if_paused()
            self.day += 1
            self.logger.info(f"===== 第 {self.day} 天 =====")

            # 夜晚阶段
            await self._night_phase()
            if self._check_winner():
                break

            # 白天阶段
            await self._day_phase()
            if self._check_winner():
                break

            # 防止死循环：超过20天强制结束
            if self.day >= 20:
                self.logger.warning("游戏超过20天，强制结束")
                break

        await self._end_game()
        self.running = False

    # ── 暂停/继续控制 ─────────────────────────────────────────────────────────

    def pause(self) -> None:
        """暂停游戏（在当前阶段完成后生效）"""
        self.paused = True
        self._step_event.clear()
        self.logger.info("游戏已暂停")

    def resume(self) -> None:
        """继续游戏"""
        self.paused = False
        self._step_event.set()
        self.logger.info("游戏已继续")

    async def _wait_if_paused(self) -> None:
        await self._step_event.wait()

    # ── 夜晚阶段 ──────────────────────────────────────────────────────────────

    async def _night_phase(self) -> None:
        self.phase = "night"
        self._emit_system(f"第{self.day}夜降临，请各位玩家闭眼。", self.day, "night")

        self._night_kill_target = None
        self._witch_heal = False
        self._witch_poison_target = None

        # 1. 狼人击杀
        await self._werewolf_action()

        # 2. 预言家查验
        await self._seer_action()

        # 3. 女巫用药
        await self._witch_action()

        # 4. 结算夜晚死亡
        await self._resolve_night_deaths()

    async def _werewolf_action(self) -> None:
        """狼人击杀"""
        wolves = [p for p in self.players.values() if p.role_type == "werewolf" and p.alive]
        if not wolves:
            return

        alive_good = [p for p in self.players.values() if p.alive and p.camp == "good"]
        if not alive_good:
            return

        gs = self._build_game_state_for(wolves[0].player_id, include_private=True)
        gs["night_instruction"] = "你是狼人，请选择今晚要击杀的目标玩家ID。"

        # 取第一只狼的决策（简化：多狼时由第一只做决策）
        decision = await wolves[0].agent.decide_night_action(gs)
        target_id = decision.get("target")
        reasoning = decision.get("reasoning", "")

        if target_id is not None and target_id in self.players and self.players[target_id].alive:
            self._night_kill_target = int(target_id)
            self.logger.info(f"[夜间] 狼人选择击杀玩家{target_id}，理由：{reasoning}")
        else:
            self.logger.info("[夜间] 狼人未选择有效目标")

        # 记录为私密对话
        self._emit_dialogue(
            speaker_id=wolves[0].player_id,
            speaker_name=wolves[0].name,
            content=f"（私密）选择击杀目标: {target_id}",
            action_type="night_action",
            target_id=self._night_kill_target,
            reasoning=reasoning,
            day=self.day, phase="night",
        )

    async def _seer_action(self) -> None:
        """预言家查验"""
        seers = [p for p in self.players.values() if p.role_type == "seer" and p.alive]
        if not seers:
            return

        seer = seers[0]
        alive_others = [p for p in self.players.values()
                        if p.alive and p.player_id != seer.player_id]
        if not alive_others:
            return

        gs = self._build_game_state_for(seer.player_id, include_private=True)
        gs["night_instruction"] = "你是预言家，请选择今晚要查验的玩家ID。"
        gs["verified"] = seer.verified

        decision = await seer.agent.decide_night_action(gs)
        target_id = decision.get("target")

        if target_id is not None and target_id in self.players:
            target = self.players[int(target_id)]
            camp_result = "狼人" if target.camp == "evil" else "好人"
            seer.verified[int(target_id)] = camp_result
            # 通知预言家查验结果（通过下一轮提示词中的 verified 字段）
            self.logger.info(f"[夜间] 预言家查验玩家{target_id}，结果：{camp_result}")

            # 更新预言家的私有上下文
            if seer.agent and seer.agent.private_context:
                seer.agent.private_context["verified_players"] = seer.verified

        self._emit_dialogue(
            speaker_id=seer.player_id,
            speaker_name=seer.name,
            content=f"（私密）查验玩家{target_id}",
            action_type="night_action",
            target_id=int(target_id) if target_id is not None else None,
            day=self.day, phase="night",
        )

    async def _witch_action(self) -> None:
        """女巫用药"""
        witches = [p for p in self.players.values() if p.role_type == "witch" and p.alive]
        if not witches:
            return

        witch = witches[0]
        gs = self._build_game_state_for(witch.player_id, include_private=True)
        gs["night_kill_target"] = self._night_kill_target
        gs["has_heal"] = witch.has_heal
        gs["has_poison"] = witch.has_poison
        gs["night_instruction"] = (
            f"你是女巫。今晚被狼人击杀的玩家是: {self._night_kill_target}。"
            f"解药: {'有' if witch.has_heal else '已用完'}，"
            f"毒药: {'有' if witch.has_poison else '已用完'}。"
            f"请决定是否用药。"
        )

        decision = await witch.agent.decide_night_action(gs)

        # 解析女巫决策（支持两种格式）
        if decision.get("action") == "witch_action":
            use_heal = decision.get("use_heal", False)
            poison_target = decision.get("poison_target")
        else:
            use_heal = decision.get("use_heal", False)
            poison_target = decision.get("poison_target") or decision.get("target")
            if decision.get("action") == "night_action":
                use_heal = False  # 如果输出的是通用格式，不触发解药

        # 解药
        if use_heal and witch.has_heal and self._night_kill_target is not None:
            witch.has_heal = False
            self._witch_heal = True
            self.logger.info(f"[夜间] 女巫使用解药救活玩家{self._night_kill_target}")

        # 毒药
        if (poison_target is not None and witch.has_poison
                and poison_target in self.players
                and self.players[int(poison_target)].alive):
            witch.has_poison = False
            self._witch_poison_target = int(poison_target)
            self.logger.info(f"[夜间] 女巫使用毒药毒死玩家{poison_target}")

        self._emit_dialogue(
            speaker_id=witch.player_id,
            speaker_name=witch.name,
            content=f"（私密）解药:{use_heal} 毒目标:{poison_target}",
            action_type="night_action",
            day=self.day, phase="night",
        )

    async def _resolve_night_deaths(self) -> None:
        """结算夜晚死亡"""
        deaths = []

        # 狼人击杀（可被解药救）
        if self._night_kill_target is not None and not self._witch_heal:
            target = self.players[self._night_kill_target]
            target.alive = False
            deaths.append((target, "night_kill"))
            self.death_records.append({
                "day": self.day, "player_id": target.player_id,
                "player_name": target.name, "role": target.role_name, "cause": "night_kill",
            })

        # 女巫毒药
        if self._witch_poison_target is not None:
            target = self.players[self._witch_poison_target]
            if target.alive:
                target.alive = False
                target.poisoned = True
                deaths.append((target, "poison"))
                self.death_records.append({
                    "day": self.day, "player_id": target.player_id,
                    "player_name": target.name, "role": target.role_name, "cause": "poison",
                })

        # 公告
        if deaths:
            for player, cause in deaths:
                cause_cn = {"night_kill": "被狼人击杀", "poison": "被女巫毒死"}.get(cause, cause)
                msg = f"昨夜，{player.name}（{player.role_name}）{cause_cn}，请出局。"
                self._emit_system(msg, self.day, "day_start")
                self.logger.info(f"[死亡] {msg}")

                # 猎人被杀（非毒死）时可以开枪
                if player.role_type == "hunter" and cause != "poison":
                    await self._hunter_shoot(player)
        else:
            self._emit_system("昨夜是平安夜，无人死亡。", self.day, "day_start")
            self.logger.info("[夜晚结算] 平安夜")

    # ── 白天阶段 ──────────────────────────────────────────────────────────────

    async def _day_phase(self) -> None:
        self.phase = "day"
        self._emit_system(f"第{self.day}天，天亮了，请各位玩家发言。", self.day, "day")

        alive_players = [p for p in self.players.values() if p.alive]

        # 每位存活玩家依次发言
        for ps in alive_players:
            await self._wait_if_paused()
            gs = self._build_game_state_for(ps.player_id)
            decision = await ps.agent.decide_speech(gs)
            content = decision.get("content", "（无发言）")
            self._emit_dialogue(
                speaker_id=ps.player_id,
                speaker_name=ps.name,
                content=content,
                action_type="speech",
                day=self.day, phase="day",
            )
            self.logger.info(f"[发言] {ps.name}: {content[:80]}")

        # 投票阶段
        await self._vote_phase()

    async def _vote_phase(self) -> None:
        self.phase = "vote"
        self._emit_system("所有人发言完毕，现在开始投票。", self.day, "vote")

        vote_counts: Counter = Counter()
        alive_players = [p for p in self.players.values() if p.alive]

        for ps in alive_players:
            await self._wait_if_paused()
            gs = self._build_game_state_for(ps.player_id)
            decision = await ps.agent.decide_vote(gs)
            target = decision.get("target")
            reasoning = decision.get("reasoning", "")

            if target is not None and int(target) in self.players and self.players[int(target)].alive:
                vote_counts[int(target)] += 1
                target_name = self.players[int(target)].name
                self._emit_dialogue(
                    speaker_id=ps.player_id,
                    speaker_name=ps.name,
                    content=f"投票给 {target_name}",
                    action_type="vote",
                    target_id=int(target),
                    reasoning=reasoning,
                    day=self.day, phase="vote",
                )
            else:
                self._emit_dialogue(
                    speaker_id=ps.player_id,
                    speaker_name=ps.name,
                    content="弃票",
                    action_type="vote",
                    day=self.day, phase="vote",
                )

        # 找出最高票
        if vote_counts:
            max_votes = max(vote_counts.values())
            candidates = [pid for pid, cnt in vote_counts.items() if cnt == max_votes]
            if len(candidates) == 1:
                voted_out = candidates[0]
            else:
                voted_out = random.choice(candidates)  # 平票随机

            target = self.players[voted_out]
            target.alive = False
            self.death_records.append({
                "day": self.day, "player_id": target.player_id,
                "player_name": target.name, "role": target.role_name, "cause": "vote",
            })
            msg = f"投票结果：{target.name}（{target.role_name}）以 {max_votes} 票被处决，请出局。"
            self._emit_system(msg, self.day, "vote")
            self.logger.info(f"[投票] {msg}")

            # 猎人被投票出局时开枪
            if target.role_type == "hunter" and not target.poisoned:
                await self._hunter_shoot(target)
        else:
            self._emit_system("本轮投票无效，无人出局。", self.day, "vote")

    async def _hunter_shoot(self, hunter: PlayerState) -> None:
        """猎人死亡后开枪"""
        if not hunter.agent:
            return

        gs = self._build_game_state_for(hunter.player_id)
        decision = await hunter.agent.decide_hunter_shoot(gs)
        target_id = decision.get("target")

        if (target_id is not None
                and int(target_id) in self.players
                and self.players[int(target_id)].alive):
            target = self.players[int(target_id)]
            target.alive = False
            self.death_records.append({
                "day": self.day, "player_id": target.player_id,
                "player_name": target.name, "role": target.role_name, "cause": "shoot",
            })
            msg = f"猎人 {hunter.name} 开枪射中了 {target.name}（{target.role_name}），一同出局！"
            self._emit_system(msg, self.day, "vote")
            self.logger.info(f"[猎人] {msg}")

    # ── 胜负判定 ──────────────────────────────────────────────────────────────

    def _check_winner(self) -> bool:
        """检查游戏是否结束"""
        alive = [p for p in self.players.values() if p.alive]
        wolves = [p for p in alive if p.role_type == "werewolf"]
        gods = [p for p in alive if p.camp == "good" and p.role_type != "villager"]
        villagers = [p for p in alive if p.role_type == "villager"]
        good = [p for p in alive if p.camp == "good"]

        # 狼人全灭 → 好人胜
        if not wolves:
            self.winner = "good"
            self.finished = True
            self._emit_system("所有狼人已被消灭！好人阵营获胜！", self.day, "end")
            self.logger.info("==== 好人阵营获胜 ====")
            return True

        # 狼人数量 >= 好人数量 → 狼人胜
        if len(wolves) >= len(good):
            self.winner = "evil"
            self.finished = True
            self._emit_system("狼人数量已超过好人！狼人阵营获胜！", self.day, "end")
            self.logger.info("==== 狼人阵营获胜 ====")
            return True

        # 所有神职死亡 → 狼人胜
        if not gods and villagers:
            self.winner = "evil"
            self.finished = True
            self._emit_system("所有神职已被消灭！狼人阵营获胜！", self.day, "end")
            self.logger.info("==== 狼人阵营获胜（神职全灭）====")
            return True

        return False

    # ── 游戏结束 ──────────────────────────────────────────────────────────────

    async def _end_game(self) -> None:
        """游戏结束，触发总结 Agent"""
        self.phase = "end"
        winner_cn = "好人阵营" if self.winner == "good" else "狼人阵营"
        self._emit_system(f"游戏结束！{winner_cn}获胜！", self.day, "end")

        self.logger.info("开始生成游戏总结...")
        summary_agent = SummaryAgent()
        players_for_summary = []
        for ps in self.players.values():
            history_text = ps.agent.get_history_text() if ps.agent else "（无记录）"
            players_for_summary.append({
                "player_name": ps.name,
                "role_name": ps.role_name,
                "role_type": ps.role_type,
                "camp": ps.camp,
                "history_text": history_text,
            })

        try:
            entries = await summary_agent.summarize_all(
                game_id=self.game_id,
                players=players_for_summary,
                winner=self.winner,
            )
            for i, (ps, entry) in enumerate(zip(self.players.values(), entries)):
                self.summaries.append(PlayerSummary(
                    player_id=ps.player_id,
                    player_name=ps.name,
                    role=ps.role_name,
                    camp=ps.camp,
                    won=(ps.camp == self.winner),
                    summary=entry.summary,
                    strategies=entry.strategies,
                    mistakes=entry.mistakes,
                    lessons=entry.lessons,
                ))
            self.logger.info(f"总结完成，共 {len(entries)} 条经验已保存")
        except Exception as e:
            self.logger.warning(f"总结生成失败: {e}")

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _build_game_state_for(self, player_id: int, include_private: bool = False) -> Dict[str, Any]:
        """
        构建某玩家视角的游戏状态。
        信息隔离：只包含该玩家有权知道的信息。
        """
        ps = self.players[player_id]
        alive_players = [
            {"id": p.player_id, "name": p.name}
            for p in self.players.values() if p.alive
        ]
        dead_players = [
            {"id": p.player_id, "name": p.name, "role": p.role_name}
            for p in self.players.values() if not p.alive
        ]
        recent_public = [
            {"speaker_name": d.speaker_name, "content": d.content}
            for d in self.dialogues[-15:]
            if d.action_type in ("speech", "system")
        ]

        gs: Dict[str, Any] = {
            "day": self.day,
            "phase": self.phase,
            "alive_players": alive_players,
            "dead_players": dead_players,
            "recent_dialogues": recent_public,
            "deaths_today": [
                d for d in self.death_records if d["day"] == self.day
            ],
            "my_id": player_id,
            "my_name": ps.name,
        }

        # 狼人可以看到同伴信息
        if include_private and ps.role_type == "werewolf":
            gs["wolf_teammates"] = [
                p.player_id for p in self.players.values()
                if p.role_type == "werewolf" and p.player_id != player_id
            ]

        # 预言家可以看到自己的查验记录
        if include_private and ps.role_type == "seer":
            gs["verified"] = ps.verified

        return gs

    def _emit_dialogue(
        self, speaker_id: int, speaker_name: str, content: str,
        action_type: str = "speech", target_id: Optional[int] = None,
        reasoning: str = "", day: int = 0, phase: str = "",
    ) -> DialogueRecord:
        record = DialogueRecord(
            day=day or self.day,
            phase=phase or self.phase,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            content=content,
            action_type=action_type,
            target_id=target_id,
            reasoning=reasoning,
        )
        self.dialogues.append(record)
        if self.on_dialogue:
            self.on_dialogue(record)
        return record

    def _emit_system(self, content: str, day: int, phase: str) -> DialogueRecord:
        return self._emit_dialogue(
            speaker_id=-1, speaker_name="🎮 主持人",
            content=content, action_type="system",
            day=day, phase=phase,
        )

    def to_game_record(self, config_name: str, player_styles: Dict[int, str]) -> GameRecord:
        """将当前游戏状态导出为 GameRecord"""
        role_assignment = {str(pid): p.role_type for pid, p in self.players.items()}
        player_names = {str(pid): p.name for pid, p in self.players.items()}
        record = GameRecord(
            game_id=self.game_id,
            start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_name=config_name,
            role_assignment=role_assignment,
            player_styles={str(k): v for k, v in player_styles.items()},
            player_names=player_names,
            dialogues=self.dialogues,
            deaths=[],
            winner=self.winner,
            summaries=self.summaries,
        )
        for d in self.death_records:
            record.add_death(
                day=d["day"], player_id=d["player_id"],
                player_name=d["player_name"], role=d["role"], cause=d["cause"],
            )
        return record
