"""API tests using a fake provider -- no API keys, network, or docling needed.

    python -m pytest tests -q
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies
from src.api.app import create_app
from src.core.config import Settings
from src.providers.base import AnswerResult, CacheProvider
from src.services import cag_service
from src.services.cag_service import CAGService


class FakeProvider(CacheProvider):
    name = "fake"

    def __init__(self):
        self.created = 0
        self.closed = 0

    def create_cache(self, document_text, system_instruction):
        self.created += 1
        return {"doc": document_text}

    def ask(self, cache_handle, question):
        return AnswerResult(
            text=f"echo: {question}",
            latency_seconds=0.01,
            input_tokens=100,
            cached_tokens=90,
            total_tokens=120,
        )

    def cache_mode(self, cache_handle):
        return "fake"

    def close(self, cache_handle):
        self.closed += 1


@pytest.fixture
def fake_service(monkeypatch):
    monkeypatch.setattr(cag_service, "load_document_text", lambda path: "one two three")
    settings = Settings(
        provider="openai",
        gemini_api_key=None,
        openai_api_key="test",
        gemini_model="g",
        openai_model="o",
        document_path=Path("policy.docx"),
    )
    return CAGService(settings, provider=FakeProvider())


@pytest.fixture
def client(fake_service):
    app = create_app()
    app.dependency_overrides[dependencies.get_cag_service] = lambda: fake_service
    with TestClient(app) as c:  # entering runs the lifespan (start/close)
        yield c


def test_health_reports_ready_after_startup(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ready"
    assert body["cache_mode"] == "fake"
    assert body["document_words"] == 3


def test_chat_returns_answer_and_stats(client):
    resp = client.post("/api/v1/chat", json={"question": "How many CL days?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "echo: How many CL days?"
    assert body["stats"]["cached_tokens"] == 90


def test_blank_question_rejected(client):
    assert client.post("/api/v1/chat", json={"question": "   "}).status_code == 422
    assert client.post("/api/v1/chat", json={"question": ""}).status_code == 422


def test_session_stats_accumulate_on_shared_instance(client):
    client.post("/api/v1/chat", json={"question": "a"})
    client.post("/api/v1/chat", json={"question": "b"})
    stats = client.get("/api/v1/stats").json()
    assert stats == {"questions_asked": 2, "cached_tokens_reused": 180, "total_tokens_billed": 240}


def test_cache_built_once_and_closed_on_shutdown(fake_service):
    app = create_app()
    app.dependency_overrides[dependencies.get_cag_service] = lambda: fake_service
    with TestClient(app) as c:
        for _ in range(3):
            c.post("/api/v1/chat", json={"question": "q"})
    assert fake_service.provider.created == 1
    assert fake_service.provider.closed == 1


def test_get_cag_service_is_singleton(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    dependencies.get_settings.cache_clear()
    dependencies.get_cag_service.cache_clear()
    try:
        assert dependencies.get_cag_service() is dependencies.get_cag_service()
    finally:
        dependencies.get_settings.cache_clear()
        dependencies.get_cag_service.cache_clear()
