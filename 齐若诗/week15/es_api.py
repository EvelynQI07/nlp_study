import yaml  # type: ignore
from elasticsearch import Elasticsearch  # type: ignore
import traceback

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

es_host = config["elasticsearch"]["host"]
es_port = config["elasticsearch"]["port"]
es_scheme = config["elasticsearch"]["scheme"]
es_username = config["elasticsearch"]["username"]
es_password = config["elasticsearch"]["password"]

if es_username != "" and es_password != "":
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
        basic_auth=(es_username, es_password)
    )
else:
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
    )

embedding_dims = config["models"]["embedding_model"][
    config["rag"]["embedding_model"]
]["dims"]


def init_es(analyzer_name: str = "ik_max_word") -> bool:
    if not es.ping():
        print("Could not connect to Elasticsearch.")
        return False

    document_meta_mapping = {
        "mappings": {
            "properties": {
                "document_name": {"type": "text", "analyzer": analyzer_name, "search_analyzer": analyzer_name},
                "abstract": {"type": "text", "analyzer": analyzer_name, "search_analyzer": analyzer_name},
                "knowledge_id": {"type": "keyword"},
                "file_path": {"type": "keyword"},
            }
        }
    }
    try:
        if not es.indices.exists(index="document_meta"):
            es.indices.create(index="document_meta", body=document_meta_mapping)
    except Exception:
        print(traceback.format_exc())
        return False

    # chunk_info 扩展：新增 chunk_type（text/image）和 image_path 字段
    chunk_info_mapping = {
        "mappings": {
            "properties": {
                "chunk_content": {"type": "text", "analyzer": analyzer_name, "search_analyzer": analyzer_name},
                "chunk_type": {"type": "keyword"},       # "text" | "image"
                "image_path": {"type": "keyword"},       # 图片存储路径（chunk_type=image 时有值）
                "knowledge_id": {"type": "keyword"},
                "document_id": {"type": "integer"},
                "page_number": {"type": "integer"},
                "chunk_id": {"type": "integer"},
                "embedding_vector": {
                    "type": "dense_vector",
                    "element_type": "float",
                    "dims": embedding_dims,
                    "index": True,
                    "index_options": {"type": "int8_hnsw"}
                }
            }
        }
    }
    try:
        if not es.indices.exists(index="chunk_info"):
            es.indices.create(index="chunk_info", body=chunk_info_mapping)
    except Exception:
        print(traceback.format_exc())
        return False

    print("Successfully connected to Elasticsearch!")
    return True


def ensure_es_initialized() -> bool:
    if init_es(analyzer_name="ik_max_word"):
        return True
    print("Analyzer `ik_max_word` init failed, falling back to `standard`...")
    return init_es(analyzer_name="standard")
