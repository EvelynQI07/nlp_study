"""角色实现

包含所有游戏角色的具体实现：
- 狼人（Werewolf）：夜晚共同选择击杀目标
- 预言家（Seer）：夜晚查验一名玩家的身份
- 女巫（Witch）：拥有解药和毒药各一瓶
- 猎人（Hunter）：死亡时可开枪带走一名玩家
- 村民（Villager）：无特殊技能，依靠推理和投票
"""
from typing import Dict, Any
from .base_role import BaseRole, RoleInfo


class WerewolfRole(BaseRole):
    """
    狼人角色

    夜晚可以与同伴商议后击杀一名玩家。
    目标：杀死所有神职或让狼人数量 >= 好人数量。
    """

    @property
    def info(self) -> RoleInfo:
        return RoleInfo(
            name="狼人",
            role_type="werewolf",
            camp="evil",
            description="邪恶阵营，夜晚共同选择击杀目标",
            win_condition="消灭所有神职角色，或使狼人数量大于等于存活好人数量",
            abilities=["夜间击杀"],
        )

    def get_night_action_description(self) -> str:
        return "你可以选择击杀一名存活的好人玩家。与同伴狼人协商后给出目标ID。"

    def get_private_context(self, player_id: int, all_players: dict) -> dict:
        ctx = super().get_private_context(player_id, all_players)
        # 狼人知道所有同伴狼人的ID
        teammates = [
            pid for pid, p in all_players.items()
            if p.get("role_type") == "werewolf" and pid != player_id
        ]
        ctx["teammates"] = teammates
        ctx["note"] = f"你的狼人同伴是: {teammates}，白天需要伪装成好人"
        return ctx


class SeerRole(BaseRole):
    """
    预言家角色

    每晚可以查验一名玩家的真实身份（好人/狼人）。
    是好人阵营最重要的神职之一。
    """

    @property
    def info(self) -> RoleInfo:
        return RoleInfo(
            name="预言家",
            role_type="seer",
            camp="good",
            description="善良阵营神职，夜晚可查验玩家身份",
            win_condition="消灭所有狼人",
            abilities=["夜间查验身份"],
        )

    def get_night_action_description(self) -> str:
        return "你可以选择查验一名玩家的身份。你将得知该玩家是好人还是狼人。"

    def get_private_context(self, player_id: int, all_players: dict) -> dict:
        ctx = super().get_private_context(player_id, all_players)
        ctx["verified_players"] = {}  # 已查验的玩家，引擎会更新
        ctx["note"] = "白天发言时，权衡是否公开你的查验结果，注意保护自己"
        return ctx


class WitchRole(BaseRole):
    """
    女巫角色

    拥有解药（复活一名当晚被杀玩家）和毒药（毒死一名玩家）各一瓶，
    每局游戏只能各使用一次。
    """

    @property
    def info(self) -> RoleInfo:
        return RoleInfo(
            name="女巫",
            role_type="witch",
            camp="good",
            description="善良阵营神职，拥有解药和毒药各一瓶",
            win_condition="消灭所有狼人",
            abilities=["解药（复活）", "毒药（毒杀）"],
        )

    def get_night_action_description(self) -> str:
        return (
            "今晚被狼人击杀的玩家已告知你。"
            "你可以选择：\n"
            "1. 使用解药救活被杀玩家（heal）\n"
            "2. 使用毒药毒死一名玩家（poison）\n"
            "3. 什么都不做（skip）\n"
            "解药和毒药每局只能各用一次。"
        )

    def get_private_context(self, player_id: int, all_players: dict) -> dict:
        ctx = super().get_private_context(player_id, all_players)
        ctx["has_heal"] = True
        ctx["has_poison"] = True
        ctx["note"] = "注意解药和毒药每局只有一瓶，谨慎使用"
        return ctx


class HunterRole(BaseRole):
    """
    猎人角色

    当猎人被狼人击杀或被投票出局时，可以选择开枪带走一名玩家。
    若被女巫毒死则无法开枪。
    """

    @property
    def info(self) -> RoleInfo:
        return RoleInfo(
            name="猎人",
            role_type="hunter",
            camp="good",
            description="善良阵营神职，死亡时可开枪带走一名玩家",
            win_condition="消灭所有狼人",
            abilities=["死亡开枪"],
        )

    def get_night_action_description(self) -> str:
        return "猎人无主动夜间行动。若被击杀，可以在死亡后立即开枪。"

    def get_private_context(self, player_id: int, all_players: dict) -> dict:
        ctx = super().get_private_context(player_id, all_players)
        ctx["note"] = "你死亡时（非毒死）可以开枪带走一名你认为是狼人的玩家，合理使用这个能力"
        return ctx


class VillagerRole(BaseRole):
    """
    村民角色

    无特殊技能，只能通过白天的发言、推理和投票来帮助好人阵营获胜。
    是好人阵营的基础力量。
    """

    @property
    def info(self) -> RoleInfo:
        return RoleInfo(
            name="村民",
            role_type="villager",
            camp="good",
            description="善良阵营，无特殊技能，依靠推理和投票",
            win_condition="消灭所有狼人",
            abilities=[],
        )

    def get_night_action_description(self) -> str:
        return "村民无夜间行动，请等待天亮。"

    def get_private_context(self, player_id: int, all_players: dict) -> dict:
        ctx = super().get_private_context(player_id, all_players)
        ctx["note"] = "你没有特殊能力，但你的发言和投票对好人阵营至关重要，请积极参与讨论"
        return ctx


# 角色注册表，用于按名称创建角色实例
ROLE_REGISTRY: Dict[str, Any] = {
    "werewolf": WerewolfRole,
    "seer": SeerRole,
    "witch": WitchRole,
    "hunter": HunterRole,
    "villager": VillagerRole,
}

# 角色中文名映射
ROLE_CN_NAMES: Dict[str, str] = {
    "werewolf": "狼人",
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "villager": "村民",
}


def create_role(role_type: str) -> BaseRole:
    """工厂函数：根据角色类型创建角色实例"""
    if role_type not in ROLE_REGISTRY:
        raise ValueError(f"未知角色类型: {role_type}，可用: {list(ROLE_REGISTRY.keys())}")
    return ROLE_REGISTRY[role_type]()
