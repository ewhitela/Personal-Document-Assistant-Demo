"""Week 10 service tests. Fake components, no models, no ollama.

Run: python -m pytest tests/test_week10_service.py -q
"""
from __future__ import annotations

import base64
import io
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))


@dataclass
class FakePassage:
    text: str = "chunk text"
    source: str = "notes.txt"
    score: float = 0.9
    chunk_id: str = "notes.txt::chunk_0"
    metadata: dict = field(default_factory=dict)


class FakeIndex:
    def __init__(self):
        self.chunks = 0
        self.reset_calls = 0

    def add_documents(self, paths):
        self.chunks += 2 * len(paths)
        return 2 * len(paths)

    def search(self, query, k):
        return [FakePassage()]

    def count(self):
        return self.chunks

    def reset(self):
        self.reset_calls += 1
        self.chunks = 0


class FakeLLM:
    def is_ready(self):
        return True

    def generate(self, prompt, system=None):
        return f"answer to: {prompt.splitlines()[-1]}"


class FakePipeline:
    SYSTEM_PROMPT = "sys"
    k = 4

    def build_prompt(self, question, passages):
        return f"ctx\nQuestion: {question}"


class FakeTranscriber:
    def transcribe(self, path):
        return "what is chromadb"


class FakeSpeaker:
    def synthesize(self, text, out_path):
        with wave.open(out_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 160)
        return out_path


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from service import app as app_module
    monkeypatch.setattr(app_module, "DOCS_DIR", tmp_path)
    comp = app_module.Components(
        index=FakeIndex(), llm=FakeLLM(), pipeline=FakePipeline(),
        transcriber=FakeTranscriber(), speaker=FakeSpeaker(),
    )
    application = app_module.create_app(components=comp)
    with TestClient(application) as c:
        c.comp = comp
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h == {"llm_ready": True, "stt_ready": True,
                 "tts_ready": True, "indexed_chunks": 0}


def test_upload_list_delete_roundtrip(client):
    files = [("files", ("a.txt", b"hello world", "text/plain")),
             ("files", ("b.md", b"# hi", "text/markdown"))]
    r = client.post("/documents", files=files)
    assert r.status_code == 200
    assert r.json()["chunks_added"] == 4

    docs = client.get("/documents").json()["documents"]
    assert docs == ["a.txt", "b.md"]

    r = client.delete("/documents/a.txt")
    assert r.status_code == 200
    assert client.comp.index.reset_calls == 1          # reset + re-add pattern
    assert client.get("/documents").json()["documents"] == ["b.md"]

    assert client.delete("/documents/missing.txt").status_code == 404


def test_upload_rejects_unsupported_type(client):
    r = client.post("/documents",
                    files=[("files", ("x.exe", b"nope", "application/x-msdownload"))])
    assert r.status_code == 400


def test_ask_returns_answer_sources_timings(client):
    r = client.post("/ask", json={"question": "what is chromadb"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("answer to:")
    assert body["sources"][0]["source"] == "notes.txt"
    t = body["timings"]
    assert set(t) == {"retrieve_s", "generate_s", "total_s"}
    assert client.post("/ask", json={"question": "  "}).status_code == 400


def test_ask_with_speech(client):
    body = client.post("/ask", json={"question": "q", "speak": True}).json()
    wav = base64.b64decode(body["audio_b64"])
    assert wav[:4] == b"RIFF"
    assert "tts_s" in body["timings"]


def test_voice_ask(client):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    r = client.post("/voice/ask", params={"speak": "true"},
                    files={"audio": ("q.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "what is chromadb"
    assert body["answer"]
    assert set(body["timings"]) == {"stt_s", "retrieve_s", "generate_s",
                                    "tts_s", "total_s"}


def test_voice_ask_503_without_transcriber(client):
    client.comp.transcriber = None
    r = client.post("/voice/ask", files={"audio": ("q.wav", b"", "audio/wav")})
    assert r.status_code == 503


def test_speak_returns_wav(client):
    r = client.post("/speak", json={"text": "hello"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"
