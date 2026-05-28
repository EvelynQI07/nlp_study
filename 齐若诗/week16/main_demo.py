"""狼人杀游戏演示入口

用法:
    python main_demo.py                       # 默认6人局
    python main_demo.py -c simple_4           # 4人局
    python main_demo.py -c big_9              # 9人局
    python main_demo.py --no-shuffle          # 固定角色顺序（调试用）
    python main_demo.py --config standard_6  # 完整参数
"""
import asyncio
import argparse
import json
import os
import random
from datetime import datetime

from engine.game_engine import GameEngine, get_role_config, shuffle_roles
from schema.game_record import GameRecord
from schema.game_logger import GameLogger


DEFAULT_STYLES = ["bold", "cautious", "balanced", "cautious", "bold", "random",
                  "balanced", "bold", "cautious"]


async def run_game(config_name: str = "standard_6", shuffle: bool = True,
                   player_styles: dict | None = None) -> GameRecord:
    game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    roles_list = get_role_config(config_name)
    n = len(roles_list)

    if shuffle:
        role_assignment = shuffle_roles(roles_list)
    else:
        role_assignment = {i: r for i, r in enumerate(roles_list)}

    if player_styles is None:
        player_styles = {i: DEFAULT_STYLES[i % len(DEFAULT_STYLES)] for i in range(n)}

    player_names = [f"玩家{i+1}" for i in range(n)]
    logger = GameLogger(game_id=game_id, log_dir="logs")

    print("=" * 60)
    print(f"🐺 狼人杀 AI 对战  配置: {config_name}")
    print("=" * 60)
    print(f"游戏ID: {game_id}")
    for pid, role in role_assignment.items():
        print(f"  {player_names[pid]}: {role}  风格:{player_styles.get(pid, 'balanced')}")
    print("=" * 60)

    engine = GameEngine(
        game_id=game_id,
        player_names=player_names,
        logger=logger,
        on_dialogue=lambda d: print(
            f"[{d.phase}] {d.speaker_name}: {d.content}"
        ) if d.action_type in ("speech", "system") else None,
    )
    engine.initialize(role_assignment, player_styles)
    await engine.start()

    record = engine.to_game_record(config_name, player_styles)
    os.makedirs("logs", exist_ok=True)
    path = record.save("logs")
    print(f"\n游戏记录已保存: {path}")
    print("\n" + "=" * 60)
    print(record.summary_text())
    print("=" * 60)

    if engine.summaries:
        print("\n玩家复盘总结:")
        for s in engine.summaries:
            won_text = "✅" if s.won else "❌"
            print(f"\n  {won_text} {s.player_name}（{s.role}）")
            print(f"     {s.summary}")
            if s.lessons:
                print(f"     教训: {s.lessons}")

    return record


def main():
    parser = argparse.ArgumentParser(description="狼人杀 AI 对战演示")
    parser.add_argument("--config", "-c", default="standard_6",
                        choices=["standard_6", "simple_4", "big_9"])
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--styles", "-s", type=str, default=None,
                        help='玩家风格 JSON，如 \'{"0":"bold","1":"cautious"}\'')
    args = parser.parse_args()

    styles = None
    if args.styles:
        try:
            raw = json.loads(args.styles)
            styles = {int(k): v for k, v in raw.items()}
        except Exception:
            print("警告: styles 格式错误，使用默认值")

    asyncio.run(run_game(
        config_name=args.config,
        shuffle=not args.no_shuffle,
        player_styles=styles,
    ))


if __name__ == "__main__":
    main()
