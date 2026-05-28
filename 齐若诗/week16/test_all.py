"""
测试套件

测试角色定义、Schema、游戏引擎核心逻辑（不调用LLM，用mock替代）。

运行：
    python -m pytest tests/ -v
"""
import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─────────────────── 角色测试 ───────────────────

from roles import create_role, ROLE_REGISTRY, ROLE_CN_NAMES


def test_all_roles_exist():
    """确保所有角色类型都可以创建"""
    for role_type in ["werewolf", "seer", "witch", "hunter", "villager"]:
        role = create_role(role_type)
        assert role.name
        assert role.camp in ("good", "evil")
        assert role.role_type == role_type


def test_werewolf_is_evil():
    role = create_role("werewolf")
    assert role.camp == "evil"


def test_seer_is_good():
    role = create_role("seer")
    assert role.camp == "good"


def test_villager_has_no_ability():
    role = create_role("villager")
    assert len(role.info.abilities) == 0


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        create_role("god")


def test_werewolf_private_context_includes_teammates():
    """狼人的私有上下文应包含同伴信息"""
    role = create_role("werewolf")
    ctx = role.get_private_context(0, {
        0: {"role_type": "werewolf"}, 1: {"role_type": "werewolf"}, 2: {"role_type": "villager"}
    })
    assert "teammates" in ctx
    assert 1 in ctx["teammates"]
    assert 2 not in ctx["teammates"]


def test_seer_private_context_has_verified():
    role = create_role("seer")
    ctx = role.get_private_context(1, {1: {"role_type": "seer"}, 2: {"role_type": "werewolf"}})
    assert "verified_players" in ctx


# ─────────────────── Schema 测试 ───────────────────

from schema.game_record import GameRecord, DialogueRecord, DeathRecord


def test_game_record_create():
    r = GameRecord(
        game_id="test_001",
        start_time="2026-01-01 00:00:00",
        config_name="simple_4",
        role_assignment={"0": "werewolf", "1": "seer", "2": "witch", "3": "villager"},
        player_styles={"0": "bold", "1": "cautious", "2": "balanced", "3": "random"},
    )
    assert r.game_id == "test_001"
    assert len(r.dialogues) == 0


def test_game_record_add_death():
    r = GameRecord(
        game_id="test_002", start_time="2026-01-01",
        config_name="simple_4",
        role_assignment={}, player_styles={},
    )
    r.add_death(1, 0, "玩家1", "狼人", "vote")
    assert len(r.deaths) == 1
    assert r.deaths[0].cause == "vote"


def test_dialogue_record_defaults():
    d = DialogueRecord(speaker_name="玩家1", content="我不是狼人")
    assert d.action_type == "speech"
    assert d.speaker_id == -1  # 默认


def test_game_record_summary_text():
    r = GameRecord(
        game_id="test_003", start_time="2026-01-01",
        config_name="simple_4",
        role_assignment={"0": "werewolf"}, player_styles={"0": "bold"},
        winner="good",
    )
    text = r.summary_text()
    assert "好人阵营" in text
    assert "test_003" in text


def test_game_record_save_load(tmp_path):
    r = GameRecord(
        game_id="test_save", start_time="2026-01-01",
        config_name="simple_4",
        role_assignment={"0": "werewolf"}, player_styles={"0": "bold"},
        winner="evil",
    )
    r.add_death(1, 0, "玩家1", "村民", "night_kill")
    path = r.save(str(tmp_path))

    loaded = GameRecord.load(path)
    assert loaded.game_id == "test_save"
    assert loaded.winner == "evil"
    assert len(loaded.deaths) == 1


# ─────────────────── 引擎测试（Mock LLM）───────────────────

from unittest.mock import AsyncMock, patch
from engine.game_engine import GameEngine, shuffle_roles, get_role_config


def test_shuffle_roles():
    roles = get_role_config("simple_4")
    assignment = shuffle_roles(roles)
    assert len(assignment) == 4
    assert set(assignment.values()) == {"werewolf", "seer", "witch", "villager"}


def test_get_role_config_invalid():
    with pytest.raises(ValueError):
        get_role_config("nonexistent")


def test_engine_initialize():
    engine = GameEngine(game_id="test_engine_001")
    assignment = {0: "werewolf", 1: "seer", 2: "witch", 3: "villager"}
    styles = {0: "bold", 1: "cautious", 2: "balanced", 3: "random"}

    with patch("agent.player_agent.anthropic.Anthropic"):
        engine.initialize(assignment, styles)

    assert len(engine.players) == 4
    assert engine.players[0].role_type == "werewolf"
    assert engine.players[0].camp == "evil"
    assert engine.players[1].role_type == "seer"
    assert engine.players[1].camp == "good"


def test_check_winner_wolves_gone():
    engine = GameEngine(game_id="test_winner_good")
    with patch("agent.player_agent.anthropic.Anthropic"):
        engine.initialize(
            {0: "werewolf", 1: "seer", 2: "villager"},
            {0: "bold", 1: "cautious", 2: "balanced"},
        )
    engine.players[0].alive = False  # 狼人死亡
    assert engine._check_winner() is True
    assert engine.winner == "good"


def test_check_winner_wolves_dominate():
    engine = GameEngine(game_id="test_winner_evil")
    with patch("agent.player_agent.anthropic.Anthropic"):
        engine.initialize(
            {0: "werewolf", 1: "seer", 2: "villager"},
            {0: "bold", 1: "cautious", 2: "balanced"},
        )
    # 让神职和村民都死
    engine.players[1].alive = False
    engine.players[2].alive = False
    assert engine._check_winner() is True
    assert engine.winner == "evil"


def test_build_game_state_isolation():
    """验证信息隔离：普通玩家看不到其他人的身份"""
    engine = GameEngine(game_id="test_isolation")
    with patch("agent.player_agent.anthropic.Anthropic"):
        engine.initialize(
            {0: "werewolf", 1: "villager"},
            {0: "bold", 1: "balanced"},
        )
    engine.day = 1
    engine.phase = "day"

    gs = engine._build_game_state_for(1)  # 村民视角
    # 村民不应该知道狼人同伴
    assert "wolf_teammates" not in gs


# ─────────────────── 记忆系统测试 ───────────────────

from memory.experience import ExperienceEntry, RoleExperienceStore


def test_experience_store_save_load(tmp_path):
    store = RoleExperienceStore(memory_dir=str(tmp_path))
    entry = ExperienceEntry(
        game_id="g001",
        role_type="seer",
        role_name="预言家",
        player_name="玩家1",
        won=True,
        winner_camp="good",
        summary="我成功找出了两只狼",
        strategies="第一天就公开查验结果",
        mistakes="差点被狼人带走",
        lessons="查验结果要在合适时机公开",
    )
    store.save(entry)
    loaded = store.load("seer")
    assert len(loaded) == 1
    assert loaded[0].game_id == "g001"
    assert loaded[0].won is True


def test_experience_prompt_generation(tmp_path):
    store = RoleExperienceStore(memory_dir=str(tmp_path))
    entry = ExperienceEntry(
        game_id="g002", role_type="werewolf", role_name="狼人",
        player_name="玩家2", won=False, winner_camp="good",
        summary="被预言家查出来了", strategies="",
        mistakes="太早暴露了", lessons="白天要低调",
    )
    store.save(entry)
    prompt = store.get_experience_prompt("werewolf")
    assert "白天要低调" in prompt
    assert "历史对局" in prompt


def test_experience_max_entries(tmp_path):
    """验证经验数量上限（最多保存20条）"""
    store = RoleExperienceStore(memory_dir=str(tmp_path))
    for i in range(25):
        store.save(ExperienceEntry(
            game_id=f"g{i:03d}", role_type="villager", role_name="村民",
            player_name="测试", won=i % 2 == 0, winner_camp="good" if i % 2 == 0 else "evil",
            summary="test", strategies="", mistakes="", lessons="",
        ))
    loaded = store.load("villager")
    assert len(loaded) <= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
