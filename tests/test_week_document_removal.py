"""Tests for batched document removal (POST /documents/remove).

    python tests/test_week_document_removal.py

Fakes throughout -- no network, no models. These check the thing that
motivated the change: removing several files costs one reindex, not one per
file, and the endpoint reports which requested filenames weren't found
without failing the whole batch.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import service.app as app_module
from service.app import Components, create_app


class FakeRAG:
    SYSTEM_PROMPT = "system"

    def __init__(self):
        self.llm = self

    def generate(self, prompt, system=None):
        return "ok"


class FakeIndex:
    def __init__(self):
        self.reset_calls = 0
        self.added = []

    def reset(self):
        self.reset_calls += 1

    def add_documents(self, paths):
        self.added.append(list(paths))
        return len(paths)

    def count(self):
        return 0

    def search(self, query, k):
        return []


class FakeAssistant:
    def __init__(self):
        self.rag = FakeRAG()

    def ready(self):
        return {"llm": True, "stt": False, "tts": False, "vision": False}


def make_client(docs_dir: Path, index: FakeIndex) -> TestClient:
    app_module.DOCS_DIR = docs_dir
    comp = Components(index=index, assistant=FakeAssistant())
    client = TestClient(create_app(comp))
    client.__enter__()  # runs lifespan startup so app.state.comp is set
    return client


def test_removing_three_files_triggers_exactly_one_reindex():
    with tempfile.TemporaryDirectory() as d:
        docs_dir = Path(d)
        for name in ("a.txt", "b.txt", "c.txt", "keep.txt"):
            (docs_dir / name).write_text("content")

        index = FakeIndex()
        client = make_client(docs_dir, index)

        resp = client.post(
            "/documents/remove",
            json={"filenames": ["a.txt", "b.txt", "c.txt"]},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert sorted(body["removed"]) == ["a.txt", "b.txt", "c.txt"]
        assert body["missing"] == []
        assert index.reset_calls == 1  # one rebuild for the whole batch
        assert not (docs_dir / "a.txt").exists()
        assert (docs_dir / "keep.txt").exists()


def test_missing_filenames_are_reported_not_fatal():
    with tempfile.TemporaryDirectory() as d:
        docs_dir = Path(d)
        (docs_dir / "a.txt").write_text("content")

        index = FakeIndex()
        client = make_client(docs_dir, index)

        resp = client.post(
            "/documents/remove", json={"filenames": ["a.txt", "ghost.txt"]}
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["removed"] == ["a.txt"]
        assert body["missing"] == ["ghost.txt"]


def test_all_missing_is_a_404():
    with tempfile.TemporaryDirectory() as d:
        docs_dir = Path(d)
        index = FakeIndex()
        client = make_client(docs_dir, index)

        resp = client.post("/documents/remove", json={"filenames": ["ghost.txt"]})

        assert resp.status_code == 404
        assert index.reset_calls == 0  # no wasted rebuild


def test_empty_filenames_is_a_400():
    with tempfile.TemporaryDirectory() as d:
        docs_dir = Path(d)
        index = FakeIndex()
        client = make_client(docs_dir, index)

        resp = client.post("/documents/remove", json={"filenames": []})

        assert resp.status_code == 400


def test_filename_is_basenamed_to_stay_inside_docs_dir():
    """A path-traversal attempt collapses to a bare filename, same guard as
    the old per-file endpoint had."""
    with tempfile.TemporaryDirectory() as d:
        docs_dir = Path(d)
        index = FakeIndex()
        client = make_client(docs_dir, index)

        resp = client.post(
            "/documents/remove", json={"filenames": ["../../etc/passwd"]}
        )

        assert resp.status_code == 404
        assert "passwd" in resp.json()["detail"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
