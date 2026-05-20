"""
多模态 RAG 核心模块。

相比 04-government-advanced-rag 的新增能力：
1. PDF 图片提取（PyMuPDF/fitz）
2. 视觉 LLM 生成图片描述（OpenAI Vision 兼容接口）
3. 图片描述作为独立 chunk 索引到 ES（chunk_type="image"）
4. 检索结果同时返回文本 chunk 和图片路径
5. 多模态对话：用户可直接传入 base64 图片参与对话
"""

import yaml  # type: ignore

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import os
import base64
import datetime
import traceback
from pathlib import Path
from typing import Union, List, Any, Dict, Optional, Tuple

import numpy as np
import pdfplumber
import fitz  # PyMuPDF，用于提取 PDF 中的嵌入图片
from openai import OpenAI

import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from es_api import es, ensure_es_initialized

device = config["device"]

EMBEDDING_MODEL_PARAMS: Dict[str, Any] = {}

# ── Prompt 模板 ───────────────────────────────────────────────────────────────

BASIC_QA_TEMPLATE = """现在的时间是{#TIME#}。你是一个专家，善于结合文字与图片资料回答用户问题。
请根据下面给出的资料（包含文字描述和图片描述）回答问题。
如果资料不足以回答，请回答"无法回答"。

资料：
{#RELATED_DOCUMENT#}

问题：{#QUESTION#}
"""

IMAGE_DESCRIBE_PROMPT = (
    "请详细描述这张图片的内容，包括图表数据、文字信息、图片主题等，"
    "描述要准确、完整，便于后续文本检索使用，使用中文回答。"
)


# ── 模型加载工具函数 ───────────────────────────────────────────────────────────

def _pick_model_path(local_url: str, hf_url: str) -> str:
    if local_url and os.path.isdir(local_url):
        return local_url
    return hf_url


def load_embedding_model(model_name: str, local_url: str, hf_url: str) -> None:
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
        path = _pick_model_path(local_url, hf_url)
        EMBEDDING_MODEL_PARAMS["embedding_model"] = SentenceTransformer(path)


def load_rerank_model(model_name: str, local_url: str, hf_url: str) -> None:
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-reranker-base"]:
        path = _pick_model_path(local_url, hf_url)
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(path)
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(path)
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        target_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        EMBEDDING_MODEL_PARAMS["rerank_model"].to(target_device)


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - chunk_overlap
    return chunks


# ── RAG 主类 ─────────────────────────────────────────────────────────────────

class RAG:
    def __init__(self) -> None:
        self.embedding_model_name = config["rag"]["embedding_model"]
        self.rerank_model_name = config["rag"]["rerank_model"]
        self.use_rerank = config["rag"]["use_rerank"]
        self.use_vision = config["rag"].get("use_vision", False)

        self.embedding_dims = config["models"]["embedding_model"][self.embedding_model_name]["dims"]
        self.chunk_size = config["rag"]["chunk_size"]
        self.chunk_overlap = config["rag"]["chunk_overlap"]
        self.chunk_candidate = config["rag"]["chunk_candidate"]
        self.image_store_dir = config["rag"].get("image_store_dir", "upload_files/images")
        self.image_min_area = config["rag"].get("image_min_area", 5000)

        Path(self.image_store_dir).mkdir(parents=True, exist_ok=True)

        self.llm_client = OpenAI(
            api_key=config["rag"]["llm_api_key"],
            base_url=config["rag"]["llm_base"],
        )
        self.llm_model = config["rag"]["llm_model"]

        # 视觉 LLM 客户端（可与文本 LLM 相同端点，也可独立）
        self.vision_client = OpenAI(
            api_key=config["rag"].get("vision_llm_api_key", config["rag"]["llm_api_key"]),
            base_url=config["rag"].get("vision_llm_base", config["rag"]["llm_base"]),
        )
        self.vision_model = config["rag"].get("vision_llm_model", config["rag"]["llm_model"])

        self._embedding_load_error: Optional[Exception] = None
        self._rerank_load_error: Optional[Exception] = None
        self._es_initialized = False

        if config["rag"]["use_embedding"]:
            try:
                cfg = config["models"]["embedding_model"][self.embedding_model_name]
                load_embedding_model(self.embedding_model_name, cfg.get("local_url", ""), cfg.get("hf_url", ""))
            except Exception as e:
                self._embedding_load_error = e

        if self.use_rerank:
            try:
                cfg = config["models"]["rerank_model"][self.rerank_model_name]
                load_rerank_model(self.rerank_model_name, cfg.get("local_url", ""), cfg.get("hf_url", ""))
            except Exception as e:
                self._rerank_load_error = e

    # ── ES 初始化 ─────────────────────────────────────────────────────────────

    def _ensure_es(self) -> None:
        if self._es_initialized:
            return
        if not ensure_es_initialized():
            raise RuntimeError("Elasticsearch is not ready.")
        self._es_initialized = True

    # ── Embedding ─────────────────────────────────────────────────────────────

    def get_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        if self.embedding_model_name in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
            if self._embedding_load_error:
                raise RuntimeError(f"Embedding model load failed: {self._embedding_load_error}")
            if "embedding_model" not in EMBEDDING_MODEL_PARAMS:
                raise RuntimeError("Embedding model is not loaded.")
            return EMBEDDING_MODEL_PARAMS["embedding_model"].encode(text, normalize_embeddings=True)
        raise NotImplementedError(f"Unsupported embedding model: {self.embedding_model_name}")

    # ── Rerank ────────────────────────────────────────────────────────────────

    def get_rank(self, text_pair: List[Tuple[str, str]]) -> np.ndarray:
        if self.rerank_model_name in ["bge-reranker-base"]:
            if self._rerank_load_error:
                raise RuntimeError(f"Rerank model load failed: {self._rerank_load_error}")
            if "rerank_model" not in EMBEDDING_MODEL_PARAMS:
                raise RuntimeError("Rerank model is not loaded.")
            with torch.no_grad():
                inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                    text_pair, padding=True, truncation=True,
                    return_tensors="pt", max_length=512,
                )
                target_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1,).float()
                return scores.data.cpu().numpy()
        raise NotImplementedError(f"Unsupported rerank model: {self.rerank_model_name}")

    # ── 视觉 LLM：图片描述 ────────────────────────────────────────────────────

    def describe_image(self, image_path: str) -> str:
        """
        调用视觉 LLM 对图片生成文字描述。
        image_path 为本地文件路径，读取后转为 base64 data URI。
        """
        if not self.use_vision:
            return ""
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = Path(image_path).suffix.lstrip(".").lower()
            mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "gif", "webp") else "image/png"
            data_uri = f"data:{mime};base64,{b64}"
            response = self.vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                            {"type": "text", "text": IMAGE_DESCRIBE_PROMPT},
                        ],
                    }
                ],
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except Exception:
            print(traceback.format_exc())
            return ""

    # ── PDF 内容提取（文本 + 图片） ────────────────────────────────────────────

    def _extract_images_from_page(
        self, fitz_doc: fitz.Document, page_number: int, document_id: int
    ) -> List[str]:
        """
        从 PDF 某页提取嵌入图片，保存到磁盘，返回图片路径列表。
        """
        page = fitz_doc[page_number]
        image_paths: List[str] = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = fitz_doc.extract_image(xref)
            w, h = base_image.get("width", 0), base_image.get("height", 0)
            if w * h < self.image_min_area:
                continue
            img_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            filename = f"doc{document_id}_p{page_number}_img{img_index}.{ext}"
            save_path = os.path.join(self.image_store_dir, filename)
            with open(save_path, "wb") as f:
                f.write(img_bytes)
            image_paths.append(save_path)
        return image_paths

    def _extract_pdf_content(
        self, knowledge_id: int, document_id: int, title: str, file_path: str
    ) -> bool:
        self._ensure_es()
        try:
            pdf_plumber = pdfplumber.open(file_path)
            fitz_doc = fitz.open(file_path)
        except Exception:
            print(traceback.format_exc())
            return False

        page_count = len(pdf_plumber.pages)
        print(f"{file_path} pages: {page_count}")

        abstract_parts: List[str] = []
        chunk_idx_global = 0

        for page_number in range(page_count):
            # ── 文本处理 ──────────────────────────────────────────────────────
            page_text = pdf_plumber.pages[page_number].extract_text() or ""
            if page_number < 3:
                abstract_parts.append(page_text)

            # 整页文本作为一个 chunk（chunk_id=0）
            if page_text.strip():
                try:
                    page_vec = self.get_embedding(page_text)
                    es.index(index="chunk_info", document={
                        "document_id": document_id,
                        "knowledge_id": str(knowledge_id),
                        "page_number": page_number,
                        "chunk_id": 0,
                        "chunk_type": "text",
                        "chunk_content": page_text,
                        "image_path": "",
                        "embedding_vector": [float(x) for x in page_vec],
                    })
                except Exception:
                    print(traceback.format_exc())

            # 滑动窗口切分文本 chunk
            text_chunks = split_text_with_overlap(page_text, self.chunk_size, self.chunk_overlap)
            if text_chunks:
                try:
                    chunk_vecs = self.get_embedding(text_chunks)
                    for ci, (chunk_text, vec) in enumerate(zip(text_chunks, chunk_vecs), start=1):
                        es.index(index="chunk_info", document={
                            "document_id": document_id,
                            "knowledge_id": str(knowledge_id),
                            "page_number": page_number,
                            "chunk_id": ci,
                            "chunk_type": "text",
                            "chunk_content": chunk_text,
                            "image_path": "",
                            "embedding_vector": [float(x) for x in vec],
                        })
                except Exception:
                    print(traceback.format_exc())

            # ── 图片处理 ──────────────────────────────────────────────────────
            img_paths = self._extract_images_from_page(fitz_doc, page_number, document_id)
            for img_path in img_paths:
                description = self.describe_image(img_path)
                if not description:
                    description = f"[图片：第{page_number + 1}页，路径：{img_path}]"
                try:
                    img_vec = self.get_embedding(description)
                    chunk_idx_global += 1
                    es.index(index="chunk_info", document={
                        "document_id": document_id,
                        "knowledge_id": str(knowledge_id),
                        "page_number": page_number,
                        "chunk_id": chunk_idx_global,
                        "chunk_type": "image",
                        "chunk_content": description,   # 存图片的文字描述，用于全文和向量检索
                        "image_path": img_path,
                        "embedding_vector": [float(x) for x in img_vec],
                    })
                except Exception:
                    print(traceback.format_exc())

        # 文档元信息
        try:
            es.index(index="document_meta", document={
                "document_id": document_id,
                "knowledge_id": str(knowledge_id),
                "document_name": title,
                "file_path": file_path,
                "abstract": "\n".join(abstract_parts),
            })
        except Exception:
            print(traceback.format_exc())

        pdf_plumber.close()
        fitz_doc.close()
        return True

    def extract_content(
        self, knowledge_id: int, document_id: int, title: str, file_type: str, file_path: str
    ) -> None:
        if "pdf" in file_type:
            self._extract_pdf_content(knowledge_id, document_id, title, file_path)
        else:
            print(f"Unsupported file_type: {file_type}")
        print(f"提取完成 document_id={document_id} file_type={file_type}")

    # ── 检索 ──────────────────────────────────────────────────────────────────

    def query_document(
        self, query: str, knowledge_id: int
    ) -> Tuple[List[Dict], List[str]]:
        """
        混合检索（BM25 + 向量 RRF），返回 (records, image_paths)。
        image_paths 去重后只保留 chunk_type=="image" 的图片路径。
        """
        self._ensure_es()

        knowledge_filter = {"term": {"knowledge_id": str(knowledge_id)}}

        # BM25 全文检索
        word_resp = es.search(
            index="chunk_info",
            body={
                "query": {
                    "bool": {
                        "must": [{"match": {"chunk_content": query}}],
                        "filter": [knowledge_filter],
                    }
                },
                "size": 50,
            },
            fields=["chunk_id", "document_id", "knowledge_id", "page_number",
                    "chunk_content", "chunk_type", "image_path"],
            source=False,
        )

        # 向量检索（KNN）
        emb_vec = self.get_embedding(query)
        knn_resp = es.search(
            index="chunk_info",
            knn={
                "field": "embedding_vector",
                "query_vector": [float(x) for x in emb_vec],
                "k": 50,
                "num_candidates": 100,
                "filter": knowledge_filter,
            },
            fields=["chunk_id", "document_id", "knowledge_id", "page_number",
                    "chunk_content", "chunk_type", "image_path"],
            source=False,
        )

        # RRF 融合
        k = 60
        fusion_score: Dict[str, float] = {}
        id2record: Dict[str, Dict] = {}

        for resp in [word_resp, knn_resp]:
            for idx, hit in enumerate(resp["hits"]["hits"]):
                _id = hit["_id"]
                fusion_score[_id] = fusion_score.get(_id, 0.0) + 1.0 / (idx + k)
                if _id not in id2record:
                    id2record[_id] = hit["fields"]

        top_ids = sorted(fusion_score, key=lambda x: fusion_score[x], reverse=True)[: self.chunk_candidate]
        top_records = [id2record[_id] for _id in top_ids]

        # Rerank（仅对文本内容做 rerank）
        if self.use_rerank and top_records:
            contents = [r.get("chunk_content", [""])[0] if isinstance(r.get("chunk_content"), list)
                        else r.get("chunk_content", "") for r in top_records]
            text_pair = [[query, c] for c in contents]
            scores = self.get_rank(text_pair)
            sorted_idx = np.argsort(scores)[::-1]
            top_records = [top_records[i] for i in sorted_idx]

        # 提取图片路径（去重）
        seen_images: List[str] = []
        for r in top_records:
            chunk_type = r.get("chunk_type", ["text"])
            chunk_type = chunk_type[0] if isinstance(chunk_type, list) else chunk_type
            img_path = r.get("image_path", [""])
            img_path = img_path[0] if isinstance(img_path, list) else img_path
            if chunk_type == "image" and img_path and img_path not in seen_images:
                seen_images.append(img_path)

        return top_records, seen_images

    # ── 对话 ──────────────────────────────────────────────────────────────────

    def chat(self, messages: List[Dict], top_p: float, temperature: float) -> Any:
        completion = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            top_p=top_p,
            temperature=temperature,
        )
        return completion.choices[0].message

    def _get_last_user_text(self, messages: List[Dict]) -> str:
        """从最后一条 user 消息中提取纯文本（兼容多模态 content 列表）。"""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # content 是 list（多模态）
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return " ".join(texts)
        return ""

    def chat_with_rag(
        self,
        knowledge_id: int,
        messages: List[Dict],
    ) -> Tuple[List[Dict], List[str]]:
        """
        多模态 RAG 对话。
        首轮（len==1）做 RAG 检索；后续轮直接多轮对话。
        返回 (updated_messages, retrieved_image_paths)。
        """
        retrieved_images: List[str] = []

        if len(messages) == 1:
            query = self._get_last_user_text(messages)
            related_records, retrieved_images = self.query_document(query, knowledge_id)

            # 拼接检索文本（文本 chunk + 图片描述）
            related_text = "\n".join(
                (r.get("chunk_content", [""])[0] if isinstance(r.get("chunk_content"), list)
                 else r.get("chunk_content", ""))
                for r in related_records
            )

            rag_prompt = (
                BASIC_QA_TEMPLATE
                .replace("{#TIME#}", str(datetime.datetime.now()))
                .replace("{#QUESTION#}", query)
                .replace("{#RELATED_DOCUMENT#}", related_text)
            )

            # 如果用户消息携带了图片，保留图片内容并附加 RAG 文本
            user_content = messages[0].get("content", "")
            if isinstance(user_content, list):
                # 多模态：在现有 content 列表追加 RAG 文本块
                rag_messages = [
                    {
                        "role": "user",
                        "content": user_content + [{"type": "text", "text": rag_prompt}],
                    }
                ]
            else:
                rag_messages = [{"role": "user", "content": rag_prompt}]

            # 决定用哪个模型（有图片时用视觉模型）
            has_image = isinstance(user_content, list) and any(
                c.get("type") == "image_url" for c in user_content
            )
            if has_image and self.use_vision:
                completion = self.vision_client.chat.completions.create(
                    model=self.vision_model,
                    messages=rag_messages,
                    top_p=0.7,
                    temperature=0.9,
                )
                answer = completion.choices[0].message.content or ""
            else:
                answer = self.chat(rag_messages, 0.7, 0.9).content or ""

            messages.append({"role": "assistant", "content": answer})
        else:
            # 后续多轮：直接走文本对话
            answer = self.chat(messages, 0.7, 0.9).content or ""
            messages.append({"role": "assistant", "content": answer})

        return messages, retrieved_images
