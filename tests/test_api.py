"""Integration tests for the FastAPI endpoints.

All external calls (YouTube, LLM, vector store) are mocked so tests
run offline and deterministically -- no API keys or network needed.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.chunking import Chunk
from app.ingestion import Segment, Transcript
from app.vectorstore import SearchHit

client = TestClient(app)


def _fake_transcript(video_id="TgF-uMvhNmg"):
    return Transcript(
        video_id=video_id,
        language="en",
        source="youtube",
        segments=[Segment(text="This is a test transcript.", start=0.0, duration=5.0)],
    )


def _fake_chunks():
    return [Chunk(index=0, text="This is a test transcript.", start=0.0, end=5.0)]


def _fake_hits():
    return [SearchHit(text="This is a test transcript.", start=0.0, score=0.95)]


class TestHealth:
    def test_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_response_has_version(self):
        r = client.get("/health")
        assert "version" in r.json()

    def test_response_has_llm_model(self):
        r = client.get("/health")
        assert "llm_model" in r.json()


class TestRoot:
    def test_returns_message(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "message" in r.json()


class TestIngest:
    @patch("app.main.get_transcript", return_value=_fake_transcript())
    @patch("app.main.has_video", return_value=False)
    @patch("app.main.chunk_transcript", return_value=_fake_chunks())
    @patch("app.main.embed_texts", return_value=[[0.1] * 384])
    @patch("app.main.add_chunks")
    def test_successful_ingest(self, mock_add, mock_embed, mock_chunk, mock_has, mock_get):
        r = client.post("/ingest", json={"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg"})
        assert r.status_code == 200
        body = r.json()
        assert body["video_id"] == "TgF-uMvhNmg"
        assert body["chunks_indexed"] == 1
        assert body["already_indexed"] is False

    @patch("app.main.get_transcript", return_value=_fake_transcript())
    @patch("app.main.has_video", return_value=True)
    def test_already_indexed(self, mock_has, mock_get):
        r = client.post("/ingest", json={"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg"})
        assert r.status_code == 200
        assert r.json()["already_indexed"] is True

    @patch("app.main.get_transcript", side_effect=ValueError("bad url"))
    def test_bad_url_returns_400(self, mock_get):
        r = client.post("/ingest", json={"url": "not-a-url"})
        assert r.status_code == 400


class TestAsk:
    @patch("app.main.has_video", return_value=True)
    @patch("app.main.embed_query", return_value=[0.1] * 384)
    @patch("app.main.search", return_value=_fake_hits())
    @patch("app.main.generate_answer", return_value="The answer is 42.")
    def test_successful_ask(self, mock_gen, mock_search, mock_embed, mock_has):
        r = client.post("/ask", json={"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg", "question": "What is it?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "The answer is 42."
        assert len(body["citations"]) == 1

    @patch("app.main.has_video", return_value=False)
    def test_video_not_indexed_returns_404(self, mock_has):
        r = client.post("/ask", json={"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg", "question": "x"})
        assert r.status_code == 404


class TestSummary:
    @patch("app.main.has_video", return_value=True)
    @patch("app.main.get_all_chunks", return_value=_fake_hits())
    @patch("app.main.summarize", return_value="Great video.")
    def test_successful_summary(self, mock_sum, mock_get, mock_has):
        r = client.post("/summary", json={"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg"})
        assert r.status_code == 200
        assert r.json()["summary"] == "Great video."

    @patch("app.main.has_video", return_value=False)
    def test_video_not_indexed_returns_404(self, mock_has):
        r = client.post("/summary", json={"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg"})
        assert r.status_code == 404


class TestQuiz:
    _QUIZ_ITEMS = [
        {"question": "What is 2+2?", "options": ["1", "2", "3", "4"], "answer": "4", "timestamp": "0:00"}
    ]

    @patch("app.main.has_video", return_value=True)
    @patch("app.main.get_all_chunks", return_value=_fake_hits())
    @patch("app.main.generate_quiz", return_value=[{"question": "What is 2+2?", "options": ["1", "2", "3", "4"], "answer": "4", "timestamp": "0:00"}])
    def test_successful_quiz(self, mock_quiz, mock_get, mock_has):
        r = client.post("/quiz", json={"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg", "num_questions": 1})
        assert r.status_code == 200
        body = r.json()
        assert body["video_id"] == "TgF-uMvhNmg"
        assert len(body["quiz"]) == 1
        assert body["quiz"][0]["answer"] == "4"

    @patch("app.main.has_video", return_value=False)
    def test_video_not_indexed_returns_404(self, mock_has):
        r = client.post("/quiz", json={"url": "https://www.youtube.com/watch?v=TgF-uMvhNmg"})
        assert r.status_code == 404
