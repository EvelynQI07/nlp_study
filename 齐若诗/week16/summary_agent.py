"""总结代理

游戏结束后，为每个玩家生成结构化复盘总结，
并将经验保存到记忆系统，用于未来同角色的游戏中提升表现。

自进化循环：对局 → SummaryAgent 分析 → 经验存入 memory → 下局 PlayerAgent 读取
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import anthropic

from memory.experience import ExperienceEntry, save_experience


_SUMMARY_SYSTEM = """你是一位狼人杀游戏的专业复盘分析师。
你会根据玩家在整局游戏中的经历，生成深度反思总结，帮助玩家在下一局中表现更好。
请始终以第一人称视角、用中文输出，格式为纯 JSON（不加代码块）。"""

_SUMMARY_TEMPLATE = """你刚刚完成了一局狼人杀游戏。请对自己的表现进行深度复盘。

## 你的身份
- 玩家名称：{player_name}
- 角色：{role_name}（{camp_cn}）

## 游戏结果
- 胜利方：{winner_cn}
- 你的阵营：{result_text}

## 你在本局的完整经历
{history}

## 输出要求
请输出如下 JSON（只输出 JSON，不要其他内容）：
{{
    "summary": "整体表现总结（2-3句话，回顾关键决策和转折点）",
    "strategies": "你使用了哪些策略？哪些有效、哪些无效？",
    "mistakes": "犯了哪些错误？有哪些地方可以做得更好？",
    "lessons": "对未来担任同一角色时最重要的1-2条建议"
}}"""


class SummaryAgent:
    """游戏复盘总结代理"""

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self._model = "claude-sonnet-4-20250514"

    async def summarize_player(
        self,
        game_id: str,
        player_name: str,
        role_name: str,
        role_type: str,
        camp: str,
        winner: Optional[str],
        player_history: str,
    ) -> ExperienceEntry:
        """
        为单个玩家生成复盘总结并保存为经验。

        Args:
            game_id: 游戏ID
            player_name: 玩家名称
            role_name: 中文角色名
            role_type: 英文角色类型
            camp: 阵营 "good"/"evil"
            winner: 胜利阵营 "good"/"evil"
            player_history: 该玩家的完整游戏历史文本

        Returns:
            保存好的经验条目
        """
        camp_cn = "善良阵营" if camp == "good" else "邪恶阵营"
        winner_cn = "好人阵营" if winner == "good" else "狼人阵营" if winner == "evil" else "平局"
        won = (camp == winner)
        result_text = "✅ 胜利！" if won else "❌ 失败。"

        prompt = _SUMMARY_TEMPLATE.format(
            player_name=player_name,
            role_name=role_name,
            camp_cn=camp_cn,
            winner_cn=winner_cn,
            result_text=result_text,
            history=player_history[:3000],  # 限制长度
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=800,
                system=_SUMMARY_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            parsed = self._parse_json(raw)
        except Exception as e:
            parsed = {
                "summary": f"总结生成失败: {e}",
                "strategies": "",
                "mistakes": "",
                "lessons": "",
            }

        entry = ExperienceEntry(
            game_id=game_id,
            role_type=role_type,
            role_name=role_name,
            player_name=player_name,
            won=won,
            winner_camp=winner,
            summary=parsed.get("summary", ""),
            strategies=parsed.get("strategies", ""),
            mistakes=parsed.get("mistakes", ""),
            lessons=parsed.get("lessons", ""),
        )
        save_experience(entry)
        return entry

    async def summarize_all(
        self,
        game_id: str,
        players: List[Dict[str, Any]],
        winner: Optional[str],
    ) -> List[ExperienceEntry]:
        """
        为所有玩家生成总结。

        Args:
            game_id: 游戏ID
            players: 玩家信息列表，每项包含
                     {player_name, role_name, role_type, camp, history_text}
            winner: 胜利阵营

        Returns:
            所有玩家的经验条目列表
        """
        entries = []
        for p in players:
            entry = await self.summarize_player(
                game_id=game_id,
                player_name=p["player_name"],
                role_name=p["role_name"],
                role_type=p["role_type"],
                camp=p["camp"],
                winner=winner,
                player_history=p.get("history_text", "（无历史记录）"),
            )
            entries.append(entry)
        return entries

    def _parse_json(self, output: str) -> Dict[str, str]:
        output = re.sub(r"```json\s*", "", output)
        output = re.sub(r"```\s*", "", output).strip()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"summary": output[:200], "strategies": "", "mistakes": "", "lessons": ""}
