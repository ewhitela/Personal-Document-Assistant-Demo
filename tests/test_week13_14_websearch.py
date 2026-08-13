"""Tests for the opt-in web-search fallback.

    python tests/test_web_search.py

Fakes throughout -- no network, no models, no API key. These check the two
things that actually matter: the fallback fires on exactly the right signal
(the verifier's abstention, nothing else), and every failure mode degrades to
the original document-grounded abstention rather than to a worse answer.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.types import Passage
from pdva.verifier import ABSTAIN
from pdva.websearch import WebResult, WebSearchError, results_to_passages
from service.app import Components
from service.web_fallback import answer_with_web_fallback, is_no_answer

EVEREST = Passage(
    "Mount Everest is 8849 metres tall.", "peaks.txt", 0.82, "peaks.txt::0"
)


class FakeLLM:
    """Returns a scripted answer per call, so a document pass and a web pass
    can be given different outputs."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def generate(self, prompt, system=None):
        # create_app's lifespan calls warmup_llm() before serving; that call
        # must not eat a scripted answer.
        if system == "Reply with one word.":
            return "ok"
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else ABSTAIN


class FakeRAG:
    SYSTEM_PROMPT = "system"

    def __init__(self, llm, k=6):
        self.llm = llm
        self.k = k

    def build_prompt(self, question, passages):
        body = "\n".join(
            f"[{i}] ({p.source}) {p.text}" for i, p in enumerate(passages, 1)
        )
        return f"{body}\n\nQuestion: {question}"


class FakeIndex:
    def __init__(self, passages=()):
        self.passages = list(passages)

    def search(self, query, k):
        return self.passages[:k]


class FakeAssistant:
    def __init__(self, rag):
        self.rag = rag


class FakeProvider:
    """Scripted search backend. `raises` and `results` are mutually exclusive."""

    def __init__(self, results=(), raises=None):
        self.results = list(results)
        self.raises = raises
        self.calls = []

    def search(self, query, max_results=5):
        self.calls.append(query)
        if self.raises is not None:
            raise self.raises
        return self.results


def make_components(index_passages, *llm_answers):
    llm = FakeLLM(*llm_answers)
    rag = FakeRAG(llm)
    return Components(index=FakeIndex(index_passages), assistant=FakeAssistant(rag))


WEB_HIT = WebResult(
    title="Mount Everest",
    url="https://www.en.wikipedia.org/wiki/Mount_Everest",
    snippet="Mount Everest is 8849 metres tall.",
    score=0.94,
)

def test_results_to_passages_maps_every_field():
    (p,) = results_to_passages([WEB_HIT])

    # Title is prepended: verify() looks for named entities in the passage
    # text, and they often appear only in the title.
    assert p.text.startswith("Mount Everest.")
    assert "8849" in p.text
    # Bare domain, so the UI source list reads cleanly next to filenames.
    assert p.source == "en.wikipedia.org"
    assert p.score == 0.94
    assert p.chunk_id == "web::https://www.en.wikipedia.org/wiki/Mount_Everest"
    assert p.metadata["origin"] == "web"


def test_results_to_passages_handles_a_malformed_url():
    (p,) = results_to_passages([WebResult("T", "not-a-url", "text", 0.1)])
    assert p.source == "web"

def test_is_no_answer_recognises_the_abstention():
    assert is_no_answer({"answer": ABSTAIN})
    assert is_no_answer({"answer": ""})
    assert is_no_answer({"answer": "x", "verification": {"abstained": True}})
    assert not is_no_answer({"answer": "Everest is 8849 metres tall."})


def test_documents_answer_means_the_web_is_never_touched():
    c = make_components([EVEREST], "Mount Everest is 8849 metres tall.")
    provider = FakeProvider(results=[WEB_HIT])

    body, timings = answer_with_web_fallback(c, "how tall is Everest?", provider)

    assert body["source_mode"] == "documents"
    assert provider.calls == []  # the point: no question leaves the device
    assert "web_search_s" not in timings
    assert body["sources"][0]["source"] == "peaks.txt"


# -- the happy fallback path ------------------------------------------------


def test_abstention_falls_back_to_the_web():
    c = make_components([], ABSTAIN, "Mount Everest is 8849 metres tall.")
    provider = FakeProvider(results=[WEB_HIT])

    body, timings = answer_with_web_fallback(c, "how tall is Everest?", provider)

    assert body["source_mode"] == "web"
    assert body["answer"] == "Mount Everest is 8849 metres tall."
    assert provider.calls == ["how tall is Everest?"]
    assert body["sources"][0]["source"] == "en.wikipedia.org"
    for key in ("web_search_s", "web_generate_s", "web_verify_s"):
        assert key in timings


def test_web_answers_are_verified_like_document_answers():
    """A fabricated entity in a web-sourced answer is stripped by the same
    verifier that guards the document path, and the request falls back to the
    document abstention rather than surfacing an ungrounded web answer."""
    c = make_components([], ABSTAIN, "The summit is Zorbulon Prime.")
    provider = FakeProvider(results=[WEB_HIT])

    body, _ = answer_with_web_fallback(c, "how tall is Everest?", provider)

    assert body["source_mode"] == "documents"
    assert body["answer"] == ABSTAIN
    assert "did not answer" in body["web_status"]


# -- failure modes ----------------------------------------------------------


def test_search_failure_keeps_the_document_abstention():
    c = make_components([], ABSTAIN)
    provider = FakeProvider(raises=WebSearchError("web search timed out after 5.0s"))

    body, timings = answer_with_web_fallback(c, "how tall is Everest?", provider)

    assert body["source_mode"] == "documents"
    assert body["answer"] == ABSTAIN
    assert "timed out" in body["web_status"]
    assert "web_search_s" in timings  # the attempt is still charged to the budget


def test_no_results_keeps_the_document_abstention():
    c = make_components([], ABSTAIN)
    provider = FakeProvider(results=[])

    body, _ = answer_with_web_fallback(c, "how tall is Everest?", provider)

    assert body["source_mode"] == "documents"
    assert body["web_status"] == "no web results"


def test_unconfigured_provider_is_a_quiet_no_op():
    from service import web_fallback

    saved = (web_fallback._provider, web_fallback._provider_checked)
    web_fallback.set_provider(None)
    try:
        c = make_components([], ABSTAIN)
        body, _ = answer_with_web_fallback(c, "how tall is Everest?")
        assert body["source_mode"] == "documents"
        assert "TAVILY_API_KEY" in body["web_status"]
    finally:
        web_fallback._provider, web_fallback._provider_checked = saved

def test_ask_endpoint_defaults_to_documents_only():
    """web=false (the default) must not even construct a provider."""
    from fastapi.testclient import TestClient

    from service import web_fallback
    from service.app import create_app

    c = make_components([EVEREST], "Mount Everest is 8849 metres tall.")
    provider = FakeProvider(results=[WEB_HIT])

    saved = (web_fallback._provider, web_fallback._provider_checked)
    web_fallback.set_provider(provider)
    try:
        with TestClient(create_app(c)) as client:
            body = client.post("/ask", json={"question": "how tall?"}).json()
        assert provider.calls == []
        assert "source_mode" not in body  # untouched document path
    finally:
        web_fallback._provider, web_fallback._provider_checked = saved


def test_ask_endpoint_honours_the_web_flag():
    from fastapi.testclient import TestClient

    from service import web_fallback
    from service.app import create_app

    c = make_components([], ABSTAIN, "Mount Everest is 8849 metres tall.")
    provider = FakeProvider(results=[WEB_HIT])

    saved = (web_fallback._provider, web_fallback._provider_checked)
    web_fallback.set_provider(provider)
    try:
        with TestClient(create_app(c)) as client:
            body = client.post(
                "/ask", json={"question": "how tall?", "web": True}
            ).json()
        assert body["source_mode"] == "web"
        assert body["answer"] == "Mount Everest is 8849 metres tall."
        # The web stages are folded into the reported total, not hidden.
        assert "web_search_s" in body["timings"]
        assert body["timings"]["total_s"] >= body["timings"]["web_generate_s"]
    finally:
        web_fallback._provider, web_fallback._provider_checked = saved


if __name__ == "__main__":
    import sys as _sys

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
    _sys.exit(1 if failures else 0)
