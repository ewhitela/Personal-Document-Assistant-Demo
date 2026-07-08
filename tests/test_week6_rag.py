"""Sanity tests for the Week 6 RAGPipeline.

Run it directly once you have implemented rag.py:

    python tests/test_week6_rag.py

These tests use a fake index and a fake LLM, so they run instantly with no
downloads and no ollama. They check the pipeline's plumbing: that it retrieves,
builds a grounded prompt that names its sources, and returns a RAGAnswer of the
right shape. (A real end-to-end check happens when you wire the real
DocumentIndex and LocalLLM together in the Week 10 service.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.types import Passage, RAGAnswer  # noqa: E402
from pdva.rag import RAGPipeline           # noqa: E402


class FakeIndex:
    """Stands in for DocumentIndex. Returns canned passages, ignores the query."""
    def __init__(self, passages):
        self._passages = passages

    def search(self, query, k=4):
        return self._passages[:k]


class FakeLLM:
    """Stands in for LocalLLM. Records the prompt and returns a canned answer."""
    DEFAULT_SYSTEM = "fake"

    def __init__(self):
        self.last_prompt = None
        self.last_system = None

    def generate(self, prompt, system=None):
        self.last_prompt = prompt
        self.last_system = system
        return "Open Settings and choose Security. Sources: passwords.txt"

    def stream(self, prompt, system=None):
        for piece in ["Open ", "Settings ", "and ", "choose ", "Security."]:
            yield piece


def _passages():
    return [
        Passage(text="To reset your password, open Settings, choose Security.",
                source="passwords.txt", score=0.91, chunk_id="passwords.txt::chunk_0"),
        Passage(text="Refunds are allowed within 30 days with a receipt.",
                source="refunds.txt", score=0.40, chunk_id="refunds.txt::chunk_0"),
    ]


def test_answer_shape():
    rag = RAGPipeline(index=FakeIndex(_passages()), llm=FakeLLM(), k=2)
    res = rag.answer("how do I reset my password?")
    assert isinstance(res, RAGAnswer), "answer() must return a RAGAnswer"
    assert res.answer.strip(), "answer text must be non-empty"
    assert res.sources == _passages()[:2], "sources must be the retrieved passages"
    assert "password" in res.prompt.lower(), "the prompt should contain the question"


def test_prompt_names_sources():
    rag = RAGPipeline(index=FakeIndex(_passages()), llm=FakeLLM(), k=2)
    prompt = rag.build_prompt("any question?", _passages())
    assert "passwords.txt" in prompt and "refunds.txt" in prompt, \
        "build_prompt must label each passage with its source filename"


def test_empty_passages_does_not_crash():
    rag = RAGPipeline(index=FakeIndex([]), llm=FakeLLM(), k=2)
    res = rag.answer("anything at all?")
    assert isinstance(res, RAGAnswer), "must return a RAGAnswer even with no passages"


def test_stream_answer():
    rag = RAGPipeline(index=FakeIndex(_passages()), llm=FakeLLM(), k=2)
    chunks = list(rag.stream_answer("how do I reset my password?"))
    assert "".join(chunks).strip(), "stream_answer should yield non-empty text"


def _run():
    results = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                r = fn()
                status = r.upper() if isinstance(r, str) and r in ("skip", "warn") else "PASS"
                results.append((status, name, ""))
            except NotImplementedError:
                results.append(("TODO", name, "not implemented yet"))
            except AssertionError as e:
                results.append(("FAIL", name, str(e)))
            except Exception as e:  # noqa: BLE001
                results.append(("ERROR", name, repr(e)))
    w = max(len(n) for _, n, _ in results)
    hard = 0
    for s, n, msg in results:
        print(f"{s:5s} {n:<{w}}  {msg}")
        if s in ("FAIL", "ERROR"):
            hard += 1
    print("\n" + ", ".join(f"{s}={sum(1 for x, _, _ in results if x == s)}"
                           for s in ["PASS", "SKIP", "WARN", "TODO", "FAIL", "ERROR"]
                           if any(x == s for x, _, _ in results)))
    raise SystemExit(1 if hard else 0)


if __name__ == "__main__":
    _run()
