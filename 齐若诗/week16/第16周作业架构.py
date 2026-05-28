#系统架构总览#
roles/          → 5种角色定义（base_role + 具体实现）
agent/
  player_agent  → Claude API驱动，完整对话历史作为记忆
  summary_agent → 游戏结束后复盘，生成结构化经验
engine/
  game_engine   → 回合引擎，信息隔离，夜间/白天/投票三阶段
schema/         → Pydantic模型，结构化日志
memory/         → JSON持久化经验库，最多保留20条
api/server.py   → FastAPI + SSE实时推流，REST控制接口
frontend/       → 单文件观战UI，暗色主题
main_demo.py    → 命令行一键对战
