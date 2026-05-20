"""
测试 FastAPI HTTP 接口（使用 TestClient，不依赖外部服务）。

运行方式（从项目根目录）：
    pytest test/test_api.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ── 知识库 CRUD ───────────────────────────────────────────────────────────────

class TestKnowledgeBase:
    def test_add_knowledge_base(self):
        resp = client.post("/v1/knowledge_base", json={"category": "政务", "title": "个人所得税知识库"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response_code"] == 200
        assert data["knowledge_id"] > 0
        assert data["title"] == "个人所得税知识库"

    def test_get_knowledge_base_exists(self):
        # 先创建
        create_resp = client.post("/v1/knowledge_base", json={"category": "政务", "title": "物业费知识库"})
        kid = create_resp.json()["knowledge_id"]

        resp = client.get(f"/v1/knowledge_base?knowledge_id={kid}&token=test")
        assert resp.status_code == 200
        assert resp.json()["response_code"] == 200
        assert resp.json()["title"] == "物业费知识库"

    def test_get_knowledge_base_not_found(self):
        resp = client.get("/v1/knowledge_base?knowledge_id=999999&token=test")
        assert resp.status_code == 200
        assert resp.json()["response_code"] == 404

    def test_delete_knowledge_base(self):
        create_resp = client.post("/v1/knowledge_base", json={"category": "测试", "title": "删除测试库"})
        kid = create_resp.json()["knowledge_id"]

        del_resp = client.delete(f"/v1/knowledge_base?knowledge_id={kid}&token=test")
        assert del_resp.json()["response_code"] == 200
        assert del_resp.json()["response_msg"] == "知识库删除成功"

        # 再次删除应返回 404
        del_resp2 = client.delete(f"/v1/knowledge_base?knowledge_id={kid}&token=test")
        assert del_resp2.json()["response_code"] == 404

    def test_full_lifecycle(self):
        """创建 → 查询 → 删除 → 确认不存在。"""
        r1 = client.post("/v1/knowledge_base", json={"category": "生命周期测试", "title": "LCTest"})
        kid = r1.json()["knowledge_id"]

        assert client.get(f"/v1/knowledge_base?knowledge_id={kid}&token=t").json()["response_code"] == 200
        assert client.delete(f"/v1/knowledge_base?knowledge_id={kid}&token=t").json()["response_code"] == 200
        assert client.get(f"/v1/knowledge_base?knowledge_id={kid}&token=t").json()["response_code"] == 404


# ── 文档 CRUD ─────────────────────────────────────────────────────────────────

class TestDocument:
    def test_get_document_not_found(self):
        resp = client.get("/v1/document?document_id=999999&token=test")
        assert resp.status_code == 200
        assert resp.json()["response_code"] == 404
        assert resp.json()["response_msg"] == "文档不存在"

    def test_delete_document_not_found(self):
        resp = client.delete("/v1/document?document_id=999999&token=test")
        assert resp.status_code == 200
        assert resp.json()["response_code"] == 404

    def test_add_document_to_nonexistent_knowledge(self):
        """上传文档到不存在的知识库，应返回失败。"""
        fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
        resp = client.post(
            "/v1/document",
            data={"knowledge_id": "999999", "title": "测试文档", "category": "pdf"},
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
        )
        assert resp.status_code == 200
        assert resp.json()["response_code"] == 404

    def test_add_document_success(self):
        """先创建知识库，再上传文档，验证接口返回成功。"""
        kb_resp = client.post("/v1/knowledge_base", json={"category": "测试", "title": "文档测试库"})
        kid = kb_resp.json()["knowledge_id"]

        fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
        resp = client.post(
            "/v1/document",
            data={"knowledge_id": str(kid), "title": "测试文档.pdf", "category": "pdf"},
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
        )
        data = resp.json()
        assert resp.status_code == 200
        assert data["response_code"] == 200
        assert data["document_id"] > 0
        assert data["knowledge_id"] == kid

    def test_get_document_after_upload(self):
        kb_resp = client.post("/v1/knowledge_base", json={"category": "测试", "title": "文档查询库"})
        kid = kb_resp.json()["knowledge_id"]

        fake_pdf = io.BytesIO(b"%PDF-1.4 fake")
        upload_resp = client.post(
            "/v1/document",
            data={"knowledge_id": str(kid), "title": "查询测试文档", "category": "pdf"},
            files={"file": ("q.pdf", fake_pdf, "application/pdf")},
        )
        doc_id = upload_resp.json()["document_id"]

        get_resp = client.get(f"/v1/document?document_id={doc_id}&token=test")
        assert get_resp.json()["response_code"] == 200
        assert get_resp.json()["title"] == "查询测试文档"

    def test_delete_document_after_upload(self):
        kb_resp = client.post("/v1/knowledge_base", json={"category": "测试", "title": "文档删除库"})
        kid = kb_resp.json()["knowledge_id"]

        fake_pdf = io.BytesIO(b"%PDF-1.4 fake")
        upload_resp = client.post(
            "/v1/document",
            data={"knowledge_id": str(kid), "title": "删除文档", "category": "pdf"},
            files={"file": ("del.pdf", fake_pdf, "application/pdf")},
        )
        doc_id = upload_resp.json()["document_id"]

        del_resp = client.delete(f"/v1/document?document_id={doc_id}&token=test")
        assert del_resp.json()["response_code"] == 200
        assert del_resp.json()["response_msg"] == "文档删除成功"

        get_resp = client.get(f"/v1/document?document_id={doc_id}&token=test")
        assert get_resp.json()["response_code"] == 404
