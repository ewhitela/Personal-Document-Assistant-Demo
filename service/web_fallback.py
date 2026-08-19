"""Opt-in web-search fallback for /ask (wraps answer_with_timings unchanged).

The document path is untouched and stays the default. This runs only when the
caller passes `web=true` and the local answer came back as the abstention.

1. Trigger. The signal for "no answer found" is the verifier's ABSTAIN
   sentence, not a retrieval-score threshold. A threshold has to be tuned per
   corpus and per embedding model, and it fires before generation, so it
   cannot see the case that matters most here: retrieval returned something
   plausible, the model generated from it, and verify() stripped the result as
   ungrounded. The ABSTAIN check catches that case and a genuinely empty
   retrieval, with no tuning. The cost is that a fallback request pays for one
   local generate pass before the web pass starts -- acceptable, since the
   local pass is the common case and the fallback is the exception.

2. Grounding. Web snippets go through `rag.build_prompt` and `verify()`
   unchanged, so the same numeric, entity, status, and inversion checks apply
   to a web answer as to a document answer. Snippets are short, so this strips
   more aggressively than it does over document chunks -- consistent with the
   project's standing preference for honest abstention over a fluent wrong
   answer.

3. Failure handling. Every failure mode -- no API key, timeout, HTTP error,
   zero results, or a web answer that itself fails verification -- returns the
   original document-grounded abstention. The fallback can never make a
   request worse than not having it.

Latency note: a fallback request pays two generate passes plus the search, so
it will normally exceed the 3-second budget. The budget applies to the
document path, which is what runs when this feature is off.

Run `python tests/test_web_search.py` after changing anything here. That test
uses fake providers, so it runs offline with no key and no network.
"""

from __future__ import annotations

import logging
import time

from pdva.verifier import ABSTAIN, verify
from pdva.websearch import (
    DEFAULT_RESULTS,
    TavilySearch,
    WebSearchError,
    WebSearchProvider,
    results_to_passages,
)

from .app import Components, _passage_dict, answer_with_timings

logger = logging.getLogger("pdva.service.web")

_provider: WebSearchProvider | None = None
_provider_checked = False


def get_provider() -> WebSearchProvider | None:
    """The process-wide provider, or None when web search is not configured.

    Returns:
        A ready WebSearchProvider, or None if no API key was set.

    Behavior:
        Built once and cached, so the key is read at first use and not on
        every request. A missing key is a normal, quiet state: the feature is
        simply unavailable and /ask reports that in the response body rather
        than raising. Note the cache means a key exported after the service
        started will not be picked up until it restarts.
    """
    global _provider, _provider_checked

    if not _provider_checked:
        _provider_checked = True
        candidate = TavilySearch()
        if candidate.is_ready():
            _provider = candidate
        else:
            logger.info("web search disabled: TAVILY_API_KEY is not set")

    return _provider


def set_provider(provider: WebSearchProvider | None) -> None:
    """Override the cached provider.

    The tests use this to inject a fake with no network. It also lets a caller
    swap in a different backend without touching this module.
    """
    global _provider, _provider_checked
    _provider = provider
    _provider_checked = True


def is_no_answer(body: dict) -> bool:
    """True when the document path found nothing worth saying.

    Three shapes count as "no answer": the exact ABSTAIN sentence, an empty
    answer, and a body the verifier marked as abstained (which covers the case
    where every sentence was stripped as ungrounded).
    """
    answer = (body.get("answer") or "").strip()
    if not answer or answer == ABSTAIN:
        return True
    return bool(body.get("verification", {}).get("abstained"))


def answer_with_web_fallback(
    c: Components,
    question: str,
    provider: WebSearchProvider | None = None,
    max_results: int = DEFAULT_RESULTS,
) -> tuple[dict, dict]:
    """Answer from the documents; if that abstains, try the web.

    Args:
        c:           the loaded Components (index + assistant).
        question:    the question text. This is the only thing sent off-device.
        provider:    a WebSearchProvider, or None to use the cached one.
        max_results: how many web results to request.

    Returns:
        The same `(body, timings)` pair as answer_with_timings, with two extra
        body keys:

            source_mode  "documents" or "web" -- where the answer came from.
            web_status   why the web path did or did not produce the answer.
                         Present only when the fallback was attempted.

        Three timing keys are added when the web path runs: `web_search_s`,
        `web_generate_s`, `web_verify_s`.

    Behavior:
        The document path always runs first and its result is returned
        untouched unless it abstained, so a question the documents answer
        never reaches the network.
    """
    body, timings = answer_with_timings(c, question)
    body["source_mode"] = "documents"

    if not is_no_answer(body):
        return body, timings

    provider = provider or get_provider()

    if provider is None:
        body["web_status"] = "unavailable: TAVILY_API_KEY is not set"
        return body, timings

    t0 = time.perf_counter()
    try:
        results = provider.search(question, max_results=max_results)
    except WebSearchError as e:
        timings["web_search_s"] = round(time.perf_counter() - t0, 3)
        logger.warning("web search failed: %s", e)
        body["web_status"] = f"search failed: {e}"
        return body, timings
    timings["web_search_s"] = round(time.perf_counter() - t0, 3)

    if not results:
        body["web_status"] = "no web results"
        return body, timings

    passages = results_to_passages(results)
    rag = c.assistant.rag

    t1 = time.perf_counter()
    text = rag.llm.generate(
        rag.build_prompt(question, passages), system=rag.SYSTEM_PROMPT
    )
    t2 = time.perf_counter()
    text, report = verify(text, passages, question)
    t3 = time.perf_counter()

    timings["web_generate_s"] = round(t2 - t1, 3)
    timings["web_verify_s"] = round(t3 - t2, 3)

    if text.strip() == ABSTAIN:
        # The web snippets didn't support an answer either. Keep the document
        # abstention rather than surfacing a second one, so the user sees one
        # clear "not found" instead of a confusing pair.
        body["web_status"] = "web results did not answer the question"
        return body, timings

    body["answer"] = text
    body["sources"] = [_passage_dict(p) for p in passages]
    body["source_mode"] = "web"
    body["web_status"] = f"answered from {len(passages)} web results"
    body["verification"] = {
        "passed": report.passed,
        "abstained": report.abstained,
        "ungrounded_numbers": report.ungrounded,
        "ungrounded_entities": report.ungrounded_entities,
        "status_stripped": report.status_stripped,
    }

    return body, timings
