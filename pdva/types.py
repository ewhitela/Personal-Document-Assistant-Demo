"""Shared data types used across Weeks 4-6.

These small dataclasses are the contract between modules. The DocumentIndex
returns Passage objects; the RAGPipeline returns a RAGAnswer. Keep them stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Passage:
    """One retrieved chunk of a document.

    Attributes:
        text:     the chunk's text content.
        source:   the filename (or document id) the chunk came from.
        score:    similarity to the query, higher means more relevant (0..1).
        chunk_id: the deterministic id of this chunk in the index.
        metadata: any extra fields stored alongside the chunk.
    """
    text: str
    source: str
    score: float
    chunk_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGAnswer:
    """The output of the RAG pipeline.

    Attributes:
        answer:  the generated answer text.
        sources: the passages the answer was grounded in (for citation/UI).
        prompt:  the final prompt sent to the LLM (kept for debugging).
    """
    answer: str
    sources: list[Passage]
    prompt: str = ""
