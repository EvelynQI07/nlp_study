"""基础角色定义

所有角色的抽象基类，定义了角色的基本属性和接口。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RoleInfo:
    """角色信息"""
    name: str           # 中文名称（如 "狼人"）
    role_type: str      # 英文类型（如 "werewolf"）
    camp: str           # 阵营：good / evil
    description: str    # 角色描述
    win_condition: str  # 胜利条件
    abilities: list     # 技能列表


class BaseRole(ABC):
    """
    基础角色抽象类

    每种角色都继承此类，实现其特定能力。
    """

    @property
    @abstractmethod
    def info(self) -> RoleInfo:
        """返回角色信息"""
        ...

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def role_type(self) -> str:
        return self.info.role_type

    @property
    def camp(self) -> str:
        return self.info.camp

    def get_night_action_description(self) -> str:
        """获取夜间行动描述（给LLM的提示词部分）"""
        return "无夜间行动。"

    def get_private_context(self, player_id: int, all_players: dict) -> dict:
        """
        获取该角色的私有信息（只有该角色自己知道的信息）

        Args:
            player_id: 当前玩家ID
            all_players: 所有玩家信息（引擎内部数据）

        Returns:
            私有信息字典
        """
        return {
            "role": self.name,
            "camp": "善良阵营" if self.camp == "good" else "邪恶阵营",
            "win_condition": self.info.win_condition,
        }
