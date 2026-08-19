"""Web search fallback: the provider protocol and one concrete backend.

This is the only module in the project that sends anything off the device, so
the boundary is deliberately narrow.

1. Only the question text leaves. Document text, filenames, and the index are
   never transmitted.

2. Nothing here runs unless the caller asks for it. The fallback is opt-in per
   request (see service/web_fallback.py) and off by default.

3. Results come back as ordinary `Passage` objects, so `build_prompt`, the LLM,
   and `verify()` are reused unchanged. A web answer is grounded and verified
   by exactly the same code as a document answer.

`WebSearchProvider` is a Protocol rather than a base class, so the tests can
pass a plain fake with a `search` method -- the same way the Week 6 tests pass
fake Index and LLM stubs to RAGPipeline.

Before you start:
    Create a key at https://tavily.com (the free tier is enough for a demo),
    then export it before starting the service:
        export TAVILY_API_KEY=tvly-...

Configuration is read from the environment, not config.py: an absent key must
degrade to "feature off", not to a broken import.

    TAVILY_API_KEY            required; without it the feature stays off
    PDVA_WEB_SEARCH_TIMEOUT   seconds, default 5.0
    PDVA_WEB_SEARCH_RESULTS   results requested per query, default 5

Run `python tests/test_web_search.py` after changing anything here. That test
uses fake providers, so it runs offline with no key and no network.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import requests

from .types import Passage

logger = logging.getLogger("pdva.websearch")

TAVILY_URL = "https://api.tavily.com/search"

DEFAULT_TIMEOUT = float(os.environ.get("PDVA_WEB_SEARCH_TIMEOUT", "5.0"))
DEFAULT_RESULTS = int(os.environ.get("PDVA_WEB_SEARCH_RESULTS", "5"))


class WebSearchError(RuntimeError):
    """Any provider-side failure: network, auth, bad payload, timeout."""


@dataclass
class WebResult:
    """One search hit, in a shape no provider is specific to.

    Attributes:
        title:    the page title.
        url:      the full result URL.
        snippet:  the text the provider judged relevant to the query.
        score:    provider relevance, higher means more relevant (0..1). 0.0
                  when the provider returns none.
        metadata: any extra fields a provider wants to carry through.
    """

    title: str
    url: str
    snippet: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class WebSearchProvider(Protocol):
    """Anything that turns a query into ranked text snippets.

    The one method the fallback needs. Implement this to swap Tavily for
    another backend without touching service/web_fallback.py.
    """

    def search(
        self, query: str, max_results: int = DEFAULT_RESULTS
    ) -> list[WebResult]: ...


class TavilySearch:
    """A thin wrapper over the Tavily search API.

    Calls the HTTP endpoint with `requests` rather than the `tavily-python`
    SDK. `requests` is already a dependency (Week 9 uses it for the remote
    vision backend), so this adds no new pin and needs no lock regeneration --
    which is where CUDA wheel drift has historically crept in on this project.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        search_depth: str = "basic",
    ) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self.timeout = timeout
        self.search_depth = search_depth

    def is_ready(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = DEFAULT_RESULTS) -> list[WebResult]:
        """Run one search and return the hits, best first.

        Args:
            query:       the question text, sent verbatim.
            max_results: how many hits to ask the provider for.

        Returns:
            A list of WebResult, ordered by the provider's own relevance.
            Hits with no snippet text are dropped -- they cannot ground an
            answer, and an empty passage only dilutes the prompt.

        Raises:
            WebSearchError on any failure: missing key, timeout, HTTP error,
            or a non-JSON body. The caller treats all of these the same way
            (fall back to the document abstention), so they share one type.

        Behavior:
            The timeout is short by design. This runs after a local
            retrieve+generate has already spent its share of the latency
            budget, so a slow provider must fail fast rather than leave the
            user waiting on a request that may not help.
        """
        if not self.api_key:
            raise WebSearchError("TAVILY_API_KEY is not set")

        payload = {
            "api_key": self.api_key,  # accepted alongside the bearer header
            "query": query,
            "max_results": max_results,
            "search_depth": self.search_depth,
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            r = requests.post(
                TAVILY_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except requests.Timeout as e:
            raise WebSearchError(f"web search timed out after {self.timeout}s") from e
        except requests.RequestException as e:
            raise WebSearchError(f"web search failed: {e}") from e
        except ValueError as e:
            raise WebSearchError("web search returned a non-JSON body") from e

        return [
            WebResult(
                title=(item.get("title") or "").strip(),
                url=(item.get("url") or "").strip(),
                snippet=(item.get("content") or "").strip(),
                score=float(item.get("score") or 0.0),
            )
            for item in data.get("results", [])
            if (item.get("content") or "").strip()
        ]


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc
    except ValueError:
        return "web"
    return host.removeprefix("www.") or "web"


def results_to_passages(results: list[WebResult]) -> list[Passage]:
    """Adapt search hits to the Passage contract every other module expects.

    Returns:
        One Passage per hit, ready to hand to `build_prompt` and `verify()`
        exactly as if it had come out of the DocumentIndex.

    Behavior:
        - The title is prepended to the snippet text. verify() requires a
          sentence's named entities to appear in a source passage, and the
          entity is very often in the title and not in the snippet body;
          leaving it out stripped otherwise-correct answers.
        - `source` is the bare domain, so the UI's source list reads
          "en.wikipedia.org" next to "lease.pdf" and the provenance of an
          answer is obvious at a glance.
        - `chunk_id` is prefixed "web::" so a web passage can never collide
          with an index chunk id.
    """
    passages = []

    for r in results:
        text = f"{r.title}. {r.snippet}" if r.title else r.snippet
        passages.append(
            Passage(
                text=text,
                source=_domain(r.url),
                score=r.score,
                chunk_id=f"web::{r.url}",
                metadata={"url": r.url, "title": r.title, "origin": "web"},
            )
        )

    return passages
