"""玩家 AI 代理

每个玩家对应一个 PlayerAgent，调用 Anthropic Claude API 进行推理和决策。
信息严格隔离：每个 Agent 只能看到属于自己视角的信息。

决策风格：
- cautious（谨慎型）：宁可放过，不轻易误杀
- bold（大胆型）：敢于冒险，主动带队
- random（随机型）：依靠直觉，不过度分析
- balanced（平衡型）：综合考虑，理性分析
"""
from __future__ import annotations

import json
import re
import os
from typing import Any, Dict, List, Optional

import anthropic

from memory.experience import get_experience_prompt

# ── 决策风格定义 ───────────────────────────────────────────────────────────────

DECISION_STYLES: Dict[str, Dict[str, str]] = {
    "cautious": {
        "name": "谨慎型",
        "description": "宁可放过可疑玩家，也不轻易误杀好人",
        "speech_tendency": "保守分析，少说少错，不主动指控他人",
        "vote_tendency": "跟大多数票，不首先提名",
        "night_tendency": "不轻易用药，查验时优先选最可疑的",
    },
    "bold": {
        "name": "大胆型",
        "description": "敢于冒险，快速做出判断，主动主导局面",
        "speech_tendency": "激进指控，主动带节奏，锁定嫌疑人",
        "vote_tendency": "果断投票，不怕误杀，认准了就投",
        "night_tendency": "果断用药/击杀，不犹豫",
    },
    "random": {
        "name": "随机型",
        "description": "不过度分析，依靠直觉和当场感受",
        "speech_tendency": "随性发言，不固定模式",
        "vote_tendency": "凭感觉投票，不一定跟票",
        "night_tendency": "随机选择目标",
    },
    "balanced": {
        "name": "平衡型",
        "description": "综合考虑各种因素，理性分析局势",
        "speech_tendency": "客观分析各方信息，给出理性判断",
        "vote_tendency": "综合发言信息后理性投票",
        "night_tendency": "合理权衡后使用能力",
    },
}


class PlayerAgent:
    """
    玩家 AI 代理

    维护该玩家的完整对话历史（包括私有信息），
    每次决策时将历史作为 messages 传给 Claude，实现"有记忆"的决策。
    """

    def __init__(
        self,
        player_id: int,
        player_name: str,
        role_name: str,
        role_type: str,
        camp: str,
        private_context: Dict[str, Any],
        decision_style: str = "balanced",
    ):
        self.player_id = player_id
        self.player_name = player_name
        self.role_name = role_name
        self.role_type = role_type
        self.camp = camp
        self.private_context = private_context
        self.decision_style = decision_style

        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self._model = "claude-sonnet-4-20250514"

        # 对话历史：[{"role": "user"/"assistant", "content": "..."}]
        self._history: List[Dict[str, str]] = []
        self._system_prompt = self._build_system_prompt()

    # ── 系统提示词构建 ────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        camp_cn = "善良阵营" if self.camp == "good" else "邪恶阵营"
        style = DECISION_STYLES.get(self.decision_style, DECISION_STYLES["balanced"])
        experience_section = get_experience_prompt(self.role_type)

        return f"""你是一个狼人杀游戏中的玩家，请完全代入你的角色进行游戏。

## 你的身份
- 玩家名称：{self.player_name}
- 角色：{self.role_name}
- 阵营：{camp_cn}

## 私有信息（只有你知道）
{json.dumps(self.private_context, ensure_ascii=False, indent=2)}

## 游戏规则
- 夜晚：狼人击杀目标；预言家查验身份；女巫决定是否用解药/毒药；猎人死亡时可开枪
- 白天：所有存活玩家公开发言，然后投票处决嫌疑人
- 好人胜利条件：消灭所有狼人
- 狼人胜利条件：消灭所有神职，或狼人数量 ≥ 存活好人数量

## 你的决策风格：{style['name']}
- 性格特点：{style['description']}
- 发言风格：{style['speech_tendency']}
- 投票倾向：{style['vote_tendency']}
- 夜间行动：{style['night_tendency']}
{experience_section}

## 输出格式要求
所有决策必须输出纯 JSON（不要加 markdown 代码块），格式：
- 夜间行动：{{"action": "night_action", "target": 玩家ID或null, "reasoning": "决策理由"}}
- 女巫专用：{{"action": "witch_action", "use_heal": true/false, "heal_target": ID或null, "use_poison": true/false, "poison_target": ID或null, "reasoning": "..."}}
- 白天发言：{{"action": "speech", "content": "你的发言内容（第一人称，不超过100字）"}}
- 投票：{{"action": "vote", "target": 玩家ID或null, "reasoning": "投票理由"}}

注意：
1. 发言时不要说出自己的真实角色（狼人要伪装）
2. target 为 null 表示放弃行动/弃票
3. 推理内容会被记录但不公开给其他玩家
"""

    # ── 核心决策接口 ──────────────────────────────────────────────────────────

    async def _call_claude(self, user_message: str) -> str:
        """调用 Claude API，保持对话历史"""
        self._history.append({"role": "user", "content": user_message})
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            system=self._system_prompt,
            messages=self._history,
        )
        assistant_content = response.content[0].text
        self._history.append({"role": "assistant", "content": assistant_content})
        return assistant_content

    async def decide_night_action(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """夜间行动决策"""
        prompt = self._build_night_prompt(game_state)
        raw = await self._call_claude(prompt)
        return self._parse_json(raw)

    async def decide_speech(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """白天发言决策"""
        prompt = self._build_speech_prompt(game_state)
        raw = await self._call_claude(prompt)
        return self._parse_json(raw)

    async def decide_vote(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """投票决策"""
        prompt = self._build_vote_prompt(game_state)
        raw = await self._call_claude(prompt)
        return self._parse_json(raw)

    async def decide_hunter_shoot(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """猎人开枪决策（死亡时触发）"""
        alive = game_state.get("alive_players", [])
        prompt = (
            f"你是猎人，刚刚死亡。现在可以开枪带走一名玩家。\n"
            f"当前存活玩家: {alive}\n"
            f"游戏对话历史已在你的记忆中。\n"
            f"请选择要开枪的目标（或不开枪），输出JSON："
            f'{{"action": "shoot", "target": 玩家ID或null, "reasoning": "理由"}}'
        )
        raw = await self._call_claude(prompt)
        return self._parse_json(raw)

    # ── Prompt 构建 ───────────────────────────────────────────────────────────

    def _build_night_prompt(self, gs: Dict[str, Any]) -> str:
        alive = gs.get("alive_players", [])
        day = gs.get("day", 1)
        recent = self._recent_dialogues(gs)
        return (
            f"【第{day}夜】夜晚开始。\n"
            f"存活玩家: {alive}\n"
            f"{recent}"
            f"\n你的角色是{self.role_name}，请决定今晚的行动。\n"
            f"（你的行动是私密的，只有主持人知道）\n"
            f"输出JSON决策："
        )

    def _build_speech_prompt(self, gs: Dict[str, Any]) -> str:
        alive = gs.get("alive_players", [])
        day = gs.get("day", 1)
        deaths_today = gs.get("deaths_today", [])
        recent = self._recent_dialogues(gs)
        death_info = f"今天死亡的玩家: {deaths_today}" if deaths_today else "今天无人死亡"
        return (
            f"【第{day}天白天】公开讨论环节。\n"
            f"{death_info}\n"
            f"存活玩家: {alive}\n"
            f"{recent}"
            f"\n现在轮到你（{self.player_name}）发言，请根据信息进行分析和表态。\n"
            f"输出JSON发言："
        )

    def _build_vote_prompt(self, gs: Dict[str, Any]) -> str:
        alive = gs.get("alive_players", [])
        day = gs.get("day", 1)
        recent = self._recent_dialogues(gs)
        return (
            f"【第{day}天投票】所有人已发言完毕，现在投票。\n"
            f"可投票目标（存活玩家）: {alive}\n"
            f"{recent}"
            f"\n请投票选出你认为最可疑的玩家（null=弃票）。\n"
            f"输出JSON投票："
        )

    def _recent_dialogues(self, gs: Dict[str, Any], n: int = 10) -> str:
        """从游戏状态中提取最近几条公开对话，作为上下文"""
        dialogues = gs.get("recent_dialogues", [])
        if not dialogues:
            return ""
        lines = ["\n近期对话记录:"]
        for d in dialogues[-n:]:
            lines.append(f"  {d.get('speaker_name', '？')}: {d.get('content', '')}")
        return "\n".join(lines) + "\n"

    # ── 输出解析 ──────────────────────────────────────────────────────────────

    def _parse_json(self, output: str) -> Dict[str, Any]:
        """从 LLM 输出中提取 JSON"""
        # 去除 markdown 代码块
        output = re.sub(r"```json\s*", "", output)
        output = re.sub(r"```\s*", "", output)
        output = output.strip()

        # 尝试直接解析
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

        # 尝试提取第一个 JSON 对象
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {"action": "unknown", "raw_output": output[:200]}

    def get_history_text(self) -> str:
        """获取该玩家的完整对话历史文本（用于总结）"""
        lines = []
        for msg in self._history:
            role = "主持人问" if msg["role"] == "user" else "我回答"
            lines.append(f"[{role}] {msg['content'][:300]}")
        return "\n".join(lines)
