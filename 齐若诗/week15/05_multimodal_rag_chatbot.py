import yaml
from typing import Union, List, Any, Dict, Optional
from pathlib import Path

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import numpy as np
import datetime
import pdfplumber
from openai import OpenAI
from anthropic import Anthropic

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

import os
import base64

device = config.get("device", "cpu")

EMBEDDING_MODEL_PARAMS: Dict[Any, Any] = {}

BASIC_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
如果问题可以从资料中获得，则请逐步回答。

资料：
{#RELATED_DOCUMENT#}


问题：{#QUESTION#}
'''

MULTIMODAL_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长分析文档内容并回答问题。
请结合提供的文本资料和图片描述来回答问题。

文本资料：
{#TEXT_DOCUMENT#}

图片描述：
{#IMAGE_DESCRIPTIONS#}

表格内容：
{#TABLE_CONTENT#}

问题：{#QUESTION#}
'''


def _pick_model_path(local_url: str, hf_url: str) -> str:
    if local_url and os.path.isdir(local_url):
        return local_url
    return hf_url


def load_embedding_model(model_name: str, local_url: str, hf_url: str) -> None:
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
        model_path = _pick_model_path(local_url=local_url, hf_url=hf_url)
        EMBEDDING_MODEL_PARAMS["embedding_model"] = SentenceTransformer(model_path)


def load_rerank_model(model_name: str, local_url: str, hf_url: str) -> None:
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-reranker-base"]:
        model_path = _pick_model_path(local_url=local_url, hf_url=hf_url)
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        model_device = device
        if model_device == "cuda" and not torch.cuda.is_available():
            model_device = "cpu"
        EMBEDDING_MODEL_PARAMS["rerank_model"].to(model_device)


def split_text_with_overlap(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + chunk_size - chunk_overlap
    return chunks


class MultimodalRAG:
    def __init__(self):
        self.embedding_model = config["rag"].get("embedding_model", "bge-small-zh-v1.5")
        self.rerank_model = config["rag"].get("rerank_model", "bge-reranker-base")
        self.use_rerank = config["rag"].get("use_rerank", False)
        self.use_multimodal = config["rag"].get("use_multimodal", False)

        self.embedding_dims = config["models"]["embedding_model"][self.embedding_model]["dims"]

        self.chunk_size = config["rag"].get("chunk_size", 512)
        self.chunk_overlap = config["rag"].get("chunk_overlap", 128)
        self.chunk_candidate = config["rag"].get("chunk_candidate", 20)

        self.client = OpenAI(
            api_key=config["rag"]["llm_api_key"],
            base_url=config["rag"]["llm_base"]
        )
        self.llm_model = config["rag"]["llm_model"]

        self.claude_client = None
        if self.use_multimodal:
            self.claude_client = Anthropic(api_key=config["rag"].get("claude_api_key", ""))

        self._embedding_load_error: Optional[Exception] = None
        self._rerank_load_error: Optional[Exception] = None
        self._es_initialized = False

        if config["rag"].get("use_embedding", True):
            try:
                model_name = self.embedding_model
                model_cfg = config["models"]["embedding_model"][model_name]
                load_embedding_model(
                    model_name=model_name,
                    local_url=model_cfg.get("local_url", ""),
                    hf_url=model_cfg.get("hf_url", ""),
                )
            except Exception as e:
                self._embedding_load_error = e

        if self.use_rerank:
            try:
                model_name = self.rerank_model
                model_cfg = config["models"]["rerank_model"][model_name]
                load_rerank_model(
                    model_name=model_name,
                    local_url=model_cfg.get("local_url", ""),
                    hf_url=model_cfg.get("hf_url", ""),
                )
            except Exception as e:
                self._rerank_load_error = e

    def _ensure_es(self) -> None:
        if self._es_initialized:
            return
        try:
            from es_api import es, ensure_es_initialized
            ok = ensure_es_initialized()
            if not ok:
                raise RuntimeError("Elasticsearch is not ready (ping/init failed).")
            self._es_initialized = True
            self.es = es
        except ImportError:
            raise RuntimeError("Elasticsearch API not available")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def describe_image(self, image_path: str) -> str:
        """
        使用Claude 3.5 Sonnet描述图片内容
        """
        if not self.claude_client:
            raise RuntimeError("Claude API not configured for multimodal")

        base64_image = self._encode_image(image_path)
        message = self.claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": "请详细描述这张图片的内容，包括图表数据、文字信息、表格内容等。"
                        }
                    ],
                }
            ],
        )
        return message.content[0].text

    def _extract_pdf_content(self, knowledge_id: int, document_id: str, title: str, file_path: str) -> bool:
        self._ensure_es()
        try:
            pdf = pdfplumber.open(file_path)
        except Exception as e:
            print(f"打开文件失败: {e}")
            return False

        print(f"{file_path} pages: {len(pdf.pages)}")

        abstract = ""
        all_tables = []

        for page_number in range(len(pdf.pages)):
            current_page = pdf.pages[page_number]
            current_page_text = current_page.extract_text() or ""

            if page_number <= 3:
                abstract += '\n' + current_page_text

            page_images = []
            for img in current_page.images:
                try:
                    image_data = img.get_image()
                    if image_data:
                        image_desc = f"图片{len(page_images)+1}: PDF第{page_number+1}页中的图片"
                        page_images.append(image_desc)
                except Exception:
                    pass

            page_tables = []
            for table in current_page.extract_tables():
                if table:
                    table_str = "\n".join([" | ".join(str(cell or "") for cell in row) for row in table])
                    page_tables.append(table_str)
                    all_tables.append({"page": page_number, "content": table_str})

            embedding_vector = self.get_embedding(current_page_text)
            page_data = {
                "document_id": document_id,
                "knowledge_id": knowledge_id,
                "page_number": page_number,
                "chunk_id": 0,
                "chunk_content": current_page_text,
                "chunk_type": "text",
                "chunk_images": page_images,
                "chunk_tables": page_tables,
                "embedding_vector": [float(x) for x in list(embedding_vector)]
            }
            self.es.index(index="chunk_info", document=page_data)

            page_chunks = split_text_with_overlap(current_page_text, self.chunk_size, self.chunk_overlap)
            embedding_vectors = self.get_embedding(page_chunks)
            for chunk_idx, chunk in enumerate(page_chunks, 1):
                chunk_data = {
                    "document_id": document_id,
                    "knowledge_id": knowledge_id,
                    "page_number": page_number,
                    "chunk_id": chunk_idx,
                    "chunk_content": chunk,
                    "chunk_type": "text",
                    "chunk_images": [],
                    "chunk_tables": [],
                    "embedding_vector": [float(x) for x in list(embedding_vectors[chunk_idx - 1])]
                }
                self.es.index(index="chunk_info", document=chunk_data)

        document_data = {
            "document_id": document_id,
            "knowledge_id": knowledge_id,
            "document_name": title,
            "file_path": file_path,
            "abstract": abstract,
            "tables": all_tables
        }
        self.es.index(index="document_meta", document=document_data)
        return True

    def _extract_image_content(self, knowledge_id: int, document_id: str, title: str, file_path: str) -> bool:
        """
        提取图片内容，使用Claude进行图片理解
        """
        if not self.use_multimodal or not self.claude_client:
            print("多模态功能未启用")
            return False

        try:
            image_description = self.describe_image(file_path)
            embedding_vector = self.get_embedding(image_description)

            chunk_data = {
                "document_id": document_id,
                "knowledge_id": knowledge_id,
                "page_number": 0,
                "chunk_id": 0,
                "chunk_content": image_description,
                "chunk_type": "image",
                "chunk_images": [file_path],
                "chunk_tables": [],
                "embedding_vector": [float(x) for x in list(embedding_vector)]
            }
            self._ensure_es()
            self.es.index(index="chunk_info", document=chunk_data)

            document_data = {
                "document_id": document_id,
                "knowledge_id": knowledge_id,
                "document_name": title,
                "file_path": file_path,
                "abstract": image_description,
                "tables": []
            }
            self.es.index(index="document_meta", document=document_data)
            return True
        except Exception as e:
            print(f"提取图片内容失败: {e}")
            return False

    def extract_content(self, knowledge_id: int, document_id: str, title: str, file_type: str, file_path: str) -> bool:
        """
        提取文档内容（支持多模态）
        """
        file_type_lower = file_type.lower()
        if "pdf" in file_type_lower:
            return self._extract_pdf_content(knowledge_id, document_id, title, file_path)
        elif any(ext in file_type_lower for ext in ["png", "jpg", "jpeg", "gif", "bmp"]):
            return self._extract_image_content(knowledge_id, document_id, title, file_path)
        elif "word" in file_type_lower:
            pass
        print("提取完成", document_id, file_type, file_path)
        return True

    def get_embedding(self, text) -> np.ndarray:
        """
        对文本进行编码（支持批量）
        """
        if self.embedding_model in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
            if self._embedding_load_error is not None:
                raise RuntimeError(f"Embedding model load failed: {self._embedding_load_error}")
            if "embedding_model" not in EMBEDDING_MODEL_PARAMS:
                raise RuntimeError("Embedding model is not loaded.")
            return EMBEDDING_MODEL_PARAMS["embedding_model"].encode(text, normalize_embeddings=True)
        raise NotImplementedError

    def get_rank(self, text_pair) -> np.ndarray:
        """
        对文本对进行重排序
        """
        if self.rerank_model in ["bge-reranker-base"]:
            if self._rerank_load_error is not None:
                raise RuntimeError(f"Rerank model load failed: {self._rerank_load_error}")
            if "rerank_model" not in EMBEDDING_MODEL_PARAMS:
                raise RuntimeError("Rerank model is not loaded.")
            with torch.no_grad():
                inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                    text_pair, padding=True, truncation=True,
                    return_tensors='pt', max_length=512,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1, ).float()
                scores = scores.data.cpu().numpy()
                return scores
        raise NotImplementedError

    def query_document(self, query: str, knowledge_id: int, multimodal: bool = False) -> List[Dict]:
        """
        多模态文档检索
        """
        self._ensure_es()

        word_search_response = self.es.search(index="chunk_info",
                                              body={
                                                  "query": {
                                                      "bool": {
                                                          "must": [
                                                              {
                                                                  "match": {
                                                                      "chunk_content": query
                                                                  }
                                                              }
                                                          ],
                                                          "filter": [
                                                              {
                                                                  "term": {
                                                                      "knowledge_id": knowledge_id
                                                                  }
                                                              }
                                                          ]
                                                      }
                                                  },
                                                  "size": 50
                                              },
                                              fields=["chunk_id", "document_id", "knowledge_id", "page_number",
                                                      "chunk_content", "chunk_type", "chunk_images", "chunk_tables"],
                                              source=False,
                                              )

        embedding_vector = self.get_embedding(query)
        knn_query = {
            "field": "embedding_vector",
            "query_vector": [float(x) for x in list(embedding_vector)],
            "k": 50,
            "num_candidates": 100,
            "filter": {
                "term": {
                    "knowledge_id": knowledge_id
                }
            }
        }
        vector_search_response = self.es.search(
            index="chunk_info", knn=knn_query,
            fields=["chunk_id", "document_id", "knowledge_id", "page_number",
                    "chunk_content", "chunk_type", "chunk_images", "chunk_tables"],
            source=False,
        )

        k = 60
        fusion_score = {}
        search_id2record = {}

        for idx, record in enumerate(word_search_response['hits']['hits']):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)
            else:
                fusion_score[_id] += 1 / (idx + k)
            if _id not in search_id2record:
                search_id2record[_id] = {k: v[0] if isinstance(v, list) else v for k, v in record["fields"].items()}

        for idx, record in enumerate(vector_search_response['hits']['hits']):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)
            else:
                fusion_score[_id] += 1 / (idx + k)
            if _id not in search_id2record:
                search_id2record[_id] = {k: v[0] if isinstance(v, list) else v for k, v in record["fields"].items()}

        sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
        sorted_records = [search_id2record[x[0]] for x in sorted_dict][:self.chunk_candidate]

        if self.use_rerank and sorted_records:
            text_pair = [[query, record.get("chunk_content", "")] for record in sorted_records]
            rerank_scores = self.get_rank(text_pair)
            rerank_idx = np.argsort(rerank_scores)[::-1]
            sorted_records = [sorted_records[x] for x in rerank_idx]

        return sorted_records

    def chat_with_rag(
            self,
            knowledge_id: int,
            messages: List[Dict],
            image_urls: List[str] = None
    ) -> List[Dict]:
        """
        RAG增强问答（支持多模态）
        """
        if image_urls is None:
            image_urls = []

        if len(messages) == 1:
            query = messages[0]["content"]
            related_records = self.query_document(query, knowledge_id, multimodal=self.use_multimodal)

            text_documents = []
            image_descriptions = []
            table_contents = []

            for record in related_records:
                content = record.get("chunk_content", "")
                chunk_type = record.get("chunk_type", "text")

                if chunk_type == "text":
                    text_documents.append(content)
                elif chunk_type == "image":
                    image_descriptions.append(content)
                if record.get("chunk_tables"):
                    table_contents.extend(record["chunk_tables"])

            for img_url in image_urls:
                if os.path.exists(img_url):
                    try:
                        desc = self.describe_image(img_url)
                        image_descriptions.append(f"用户上传图片: {desc}")
                    except Exception as e:
                        print(f"图片描述失败: {e}")

            text_document = '\n'.join(text_documents)
            image_description = '\n'.join(image_descriptions)
            table_content = '\n'.join(table_contents)

            if image_descriptions or table_contents:
                rag_query = MULTIMODAL_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                    .replace("{#QUESTION#}", query) \
                    .replace("{#TEXT_DOCUMENT#}", text_document) \
                    .replace("{#IMAGE_DESCRIPTIONS#}", image_description) \
                    .replace("{#TABLE_CONTENT#}", table_content)
            else:
                rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                    .replace("{#QUESTION#}", query) \
                    .replace("{#RELATED_DOCUMENT#}", text_document)

            rag_response = self.chat(
                [{"role": "user", "content": rag_query}],
                0.7, 0.9
            ).content
            messages.append({"role": "assistant", "content": rag_response})
        else:
            normal_response = self.chat(
                messages,
                0.7, 0.9
            ).content
            messages.append({"role": "assistant", "content": normal_response})

        return messages

    def chat(self, messages: List[Dict], top_p: float, temperature: float) -> Any:
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            top_p=top_p,
            temperature=temperature
        )
        return completion.choices[0].message

    def query_parse(self, query: str) -> str:
        return ""

    def query_rewrite(self, query: str) -> str:
        return ""

    def batch_extract(self, knowledge_id: int, documents: List[Dict]) -> bool:
        """
        批量提取文档内容
        """
        success_count = 0
        for doc in documents:
            try:
                doc_id = doc.get("document_id")
                title = doc.get("title", "")
                file_type = doc.get("file_type", "")
                file_path = doc.get("file_path", "")

                if self.extract_content(knowledge_id, doc_id, title, file_type, file_path):
                    success_count += 1
            except Exception as e:
                print(f"处理文档失败 {doc.get('document_id')}: {e}")

        print(f"批量处理完成: {success_count}/{len(documents)}")
        return success_count == len(documents)

    def get_document_metadata(self, document_id: str) -> Optional[Dict]:
        """
        获取文档元数据
        """
        try:
            self._ensure_es()
            response = self.es.search(
                index="document_meta",
                body={"query": {"term": {"document_id": document_id}}},
                size=1
            )
            if response['hits']['hits']:
                return response['hits']['hits'][0]['_source']
            return None
        except Exception as e:
            print(f"获取文档元数据失败: {e}")
            return None