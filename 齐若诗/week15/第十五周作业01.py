"""
FastAPI 入口：多模态 RAG Chatbot。

接口列表：
  GET    /v1/knowledge_base   查询知识库
  POST   /v1/knowledge_base   新增知识库
  DELETE /v1/knowledge_base   删除知识库
  GET    /v1/document         查询文档
  POST   /v1/document         上传文档（后台解析，支持多模态提取）
  DELETE /v1/document         删除文档
  POST   /v1/embedding        文本向量化
  POST   /v1/rerank           文本重排序
  POST   /chat                多模态 RAG 对话（支持图片 + 文本输入）
"""

import yaml  # type: ignore

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import time
import uuid
import datetime
import traceback

import uvicorn
from typing_extensions import Annotated
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks

from router_schemas import (
    EmbeddingRequest, EmbeddingResponse,
    RerankRequest, RerankResponse,
    KnowledgeRequest, KnowledgeResponse,
    DocumentResponse,
    RAGRequest, RAGResponse,
)
from rag_api import RAG
from db_api import KnowledgeDocument, KnowledgeDatabase, Session

import numpy as np

app = FastAPI(title="Multimodal RAG Chatbot", version="1.0.0")


# ── 知识库管理 ────────────────────────────────────────────────────────────────

@app.get("/v1/knowledge_base")
def get_knowledge_base(knowledge_id: int, token: str) -> KnowledgeResponse:
    start = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.knowledge_id == knowledge_id
            ).first()
            if record:
                return KnowledgeResponse(
                    request_id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    title=str(record.title),
                    category=str(record.category),
                    response_code=200,
                    response_msg="知识库查询成功",
                    process_status="completed",
                    processing_time=time.time() - start,
                )
    except Exception:
        print(traceback.format_exc())

    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        title="",
        category="",
        response_code=404,
        response_msg="知识库不存在",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.post("/v1/knowledge_base")
def add_knowledge_base(req: KnowledgeRequest) -> KnowledgeResponse:
    start = time.time()
    try:
        with Session() as session:
            record = KnowledgeDatabase(
                title=req.title,
                category=req.category,
                create_dt=datetime.datetime.now(),
                update_dt=datetime.datetime.now(),
            )
            session.add(record)
            session.flush()
            knowledge_id = record.knowledge_id
            session.commit()
        return KnowledgeResponse(
            request_id=str(uuid.uuid4()),
            knowledge_id=knowledge_id,
            title=req.title,
            category=req.category,
            response_code=200,
            response_msg="知识库插入成功",
            process_status="completed",
            processing_time=time.time() - start,
        )
    except Exception:
        print(traceback.format_exc())

    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=0,
        title="",
        category="",
        response_code=504,
        response_msg="知识库插入失败",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.delete("/v1/knowledge_base")
def delete_knowledge_base(knowledge_id: int, token: str) -> KnowledgeResponse:
    start = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.knowledge_id == knowledge_id
            ).first()
            if record:
                title, category = str(record.title), str(record.category)
                session.delete(record)
                session.commit()
                return KnowledgeResponse(
                    request_id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    title=title,
                    category=category,
                    response_code=200,
                    response_msg="知识库删除成功",
                    process_status="completed",
                    processing_time=time.time() - start,
                )
    except Exception:
        print(traceback.format_exc())

    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        title="",
        category="",
        response_code=404,
        response_msg="知识库不存在",
        process_status="completed",
        processing_time=time.time() - start,
    )


# ── 文档管理 ──────────────────────────────────────────────────────────────────

@app.get("/v1/document")
def get_document(document_id: int, token: str) -> DocumentResponse:
    start = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.document_id == document_id
            ).first()
            if record:
                return DocumentResponse(
                    request_id=str(uuid.uuid4()),
                    document_id=document_id,
                    category=str(record.category),
                    title=str(record.title),
                    knowledge_id=record.knowledge_id,
                    file_type=str(record.file_type),
                    response_code=200,
                    response_msg="文档查询成功",
                    process_status="completed",
                    processing_time=time.time() - start,
                )
    except Exception:
        print(traceback.format_exc())

    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        category="",
        title="",
        knowledge_id=0,
        file_type="",
        response_code=404,
        response_msg="文档不存在",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.delete("/v1/document")
def delete_document(document_id: int, token: str) -> DocumentResponse:
    start = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.document_id == document_id
            ).first()
            if record:
                data = dict(
                    category=str(record.category),
                    title=str(record.title),
                    knowledge_id=record.knowledge_id,
                    file_type=str(record.file_type),
                )
                session.delete(record)
                session.commit()
                # TODO: 同步删除 ES 中对应 document_id 的 chunk 记录
                return DocumentResponse(
                    request_id=str(uuid.uuid4()),
                    document_id=document_id,
                    response_code=200,
                    response_msg="文档删除成功",
                    process_status="completed",
                    processing_time=time.time() - start,
                    **data,
                )
    except Exception:
        print(traceback.format_exc())

    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        category="",
        title="",
        knowledge_id=0,
        file_type="",
        response_code=404,
        response_msg="文档不存在",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.post("/v1/document")
async def add_document(
    knowledge_id: int = Form(...),
    title: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> DocumentResponse:
    start = time.time()
    response_msg = "新增文档失败"
    try:
        with Session() as session:
            kb = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.knowledge_id == knowledge_id
            ).first()
            if kb is None:
                response_msg = "知识库不存在，请提前创建"
                raise ValueError(response_msg)

            doc = KnowledgeDocument(
                title=title,
                category=category,
                knowledge_id=knowledge_id,
                file_path="",
                file_type=file.content_type,
                create_dt=datetime.datetime.now(),
                update_dt=datetime.datetime.now(),
            )
            session.add(doc)
            session.flush()
            document_id = doc.document_id
            session.commit()

        file_path = f"upload_files/document_id_{document_id}_{file.filename}"
        with open(file_path, "wb") as buf:
            buf.write(await file.read())

        with Session() as session:
            doc = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.document_id == document_id
            ).first()
            doc.file_path = file_path
            session.commit()

        background_tasks.add_task(
            RAG().extract_content,
            knowledge_id=knowledge_id,
            document_id=document_id,
            title=title,
            file_type=file.content_type,
            file_path=file_path,
        )

        return DocumentResponse(
            request_id=str(uuid.uuid4()),
            document_id=document_id,
            category=category,
            title=title,
            knowledge_id=knowledge_id,
            file_type=file.content_type,
            response_code=200,
            response_msg="文档添加成功，后台解析中",
            process_status="processing",
            processing_time=time.time() - start,
        )
    except Exception:
        print(traceback.format_exc())

    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=0,
        category="",
        title="",
        knowledge_id=0,
        file_type="",
        response_code=404,
        response_msg=response_msg,
        process_status="failed",
        processing_time=time.time() - start,
    )


# ── Embedding / Rerank ────────────────────────────────────────────────────────

@app.post("/v1/embedding")
async def semantic_embedding(req: EmbeddingRequest) -> EmbeddingResponse:
    start = time.time()
    text = [req.text] if isinstance(req.text, str) else req.text
    vector: np.ndarray = RAG().get_embedding(text)
    return EmbeddingResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.post("/v1/rerank")
async def semantic_rerank(req: RerankRequest) -> RerankResponse:
    start = time.time()
    vector: np.ndarray = RAG().get_rank(req.text_pair)
    return RerankResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start,
    )


# ── 多模态 RAG 对话 ───────────────────────────────────────────────────────────

@app.post("/chat")
def chat(req: RAGRequest) -> RAGResponse:
    """
    多模态 RAG 对话接口。

    req.message 格式（兼容 OpenAI Chat）：
    [
      {
        "role": "user",
        "content": "文字问题"          # 纯文本
      }
    ]
    或：
    [
      {
        "role": "user",
        "content": [                   # 多模态（文字 + 图片）
          {"type": "text", "text": "这张图里有什么？"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ]
    """
    start = time.time()
    # 将 Pydantic 模型转为普通 dict 列表，便于传给 OpenAI SDK
    raw_messages = [m.model_dump(exclude_none=True) for m in req.message]

    updated_messages, retrieved_images = RAG().chat_with_rag(
        req.knowledge_id, raw_messages
    )
    return RAGResponse(
        request_id=str(uuid.uuid4()),
        message=updated_messages,
        retrieved_images=retrieved_images,
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["rag"]["port"], workers=1)
