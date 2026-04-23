（1）什么是前后端分离
前后端分离 是一种软件架构模式，将应用程序分为两个独立的部分：
前端负责用户界面展示和交互，通过HTTP请求调用后端API ，后端负责业务逻辑、数据处理和存储 FastAPI服务（ main_server.py ）
本项目的前后端分离架构：
①　后端服务 （ main_server.py ）：基于 FastAPI 框架，提供 RESTful API 接口
②　路由层（ routers/ ）：定义 API 端点，处理 HTTP 请求
③　服务层（ services/ ）：实现核心业务逻辑（如对话管理、股票数据查询）
④　数据层（ models/orm.py ）：使用 SQLAlchemy ORM 操作 SQLite 数据库
⑤　前端交互 ：前端通过 HTTP 请求与后端通信
⑥　聊天接口： POST /v1/chat/ - 流式响应（SSE）
⑦　会话管理： POST /v1/chat/init 、 /get 、 /list 、 /delete
⑧　数据格式：JSON 请求体，使用 RequestForChat 和 BasicResponse 数据模型
（2）历史多轮对话的存储与使用
存储机制，项目采用双层存储架构 ：
 1. 业务数据库存储（ ./assert/sever.db ）
使用两个关联表存储对话：
models/orm.py
class ChatSessionTable(Base):
    """会话元数据表 - 存储对话的基础信息"""
    __tablename__ = 'chat_session'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey
    ('user.id'))  # 关联用户
    session_id = Column(String)  # 会话唯一
    标识
    title = Column(String(100))  # 对话标题
    start_time = Column(DateTime)
class ChatMessageTable(Base):
    """消息记录表 - 存储每条对话消息"""
    __tablename__ = 'chat_message'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey
    ('chat_session.id'))  # 关联会话
    role = Column(String(10))  # 'user' / 
    'assistant' / 'system'
    content = Column(Text)  # 消息内容
    create_time = Column(DateTime)
``` 2. Agent 会话存储（ ./assert/conversations.db ）
使用 AdvancedSQLiteSession 专门为大模型 Agent 管理对话状态：

```
# services/chat.py:142-147
session = AdvancedSQLiteSession(
    session_id=session_id,  # 与业务数据库的
    会话ID关联
    db_path="./assert/conversations.db",
    create_tables=True
)
```
历史对话作为大模型输入的流程
```
用户提问 → 检查会话 → 加载历史 → 调用大模型 → 保存响应
     ↓           ↓          ↓           ↓           ↓
  请求内容   session_id  从DB读取   构造Prompt   写入DB
1. 初始化/检查会话 ：
if session_id:
    # 检查会话是否存在，不存在则初始化
    record = session.query
    (ChatSessionTable).filter(
        ChatSessionTable.session_id == 
        session_id
    ).first()
    if not record:
        init_chat_session(user_name, 
        content, session_id, task)
```
2. 保存用户消息到数据库 ：
```
append_message2db(session_id, "user", 
content)
```
3. 加载历史对话给大模型 ：
```
# AdvancedSQLiteSession 自动加载该 
session_id 的历史消息
result = Runner.run_streamed(agent, 
input=content, session=session)
```
4. 保存助手响应 ：
```
async for event in result.stream_events():
    if isinstance(event.data, 
    ResponseTextDeltaEvent):
        yield f"{event.data.delta}"  # 流
        式返回给前端
        assistant_message += event.data.
        delta
对话结束后保存
append_message2db(session_id, 
"assistant", assistant_message)
关键设计要点
（1）会话标识：使用 session_id 作为跨请求的会话唯一标识 
（2）双层存储：业务库用于用户管理和查询，Agent库专门优化大模型对话格式。
（3）流式响应：使用 SSE（Server-Sent Events）实时推送大模型响应。
（4）角色区分 通过 role 字段区分 system 、 user 、 assistant 消息
获取历史对话
```
# services/chat.py:225-243
def get_chat_sessions(session_id: str) -> 
List[Dict[str, Any]]:
    with SessionLocal() as session:
        chat_messages = session.query
        (ChatMessageTable) \
            .join(ChatSessionTable) \
            .filter(ChatSessionTable.
            session_id == session_id) \
            .all()
        # 返回格式化的消息列表
        return [{
            "id": record.id, "role": 
            record.role, 
            "content": record.content, 
            "create_time": record.
            create_time
        } for record in chat_messages]
```
