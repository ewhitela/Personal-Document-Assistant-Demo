"""Week 6: Retrieval-augmented generation. The brain of the assistant.

Combines Week 4 (DocumentIndex) and Week 5 (LocalLLM): retrieve the most
relevant passages for a question, build a grounded prompt, and generate an
answer that uses only those passages.

You implement every method marked `raise NotImplementedError`.

Run `python -m tests.test_week6_rag` after implementing. That test uses small
fake Index and LLM stubs, so it runs without ollama and checks your pipeline
logic directly.
"""
from __future__ import annotations

from typing import Iterator

from .types import Passage, RAGAnswer
from .embedding_index import DocumentIndex
from .llm import LocalLLM
from . import config


class RAGPipeline:
    """Retrieve, then generate a grounded answer."""

    SYSTEM_PROMPT = (
        "You are a precise assistant for a personal document collection. "
        "Answer the question using ONLY the context passages provided, directly "
        "and completely in 2-4 sentences, including the key specifics from the "
        "context. If the context does not contain the answer, reply exactly: "
        "\"I don't know based on your documents.\""
        "If the context only partially answers, give the partial answer plainly without describing what the context lacks."
    )

    def __init__(self, index: DocumentIndex, llm: LocalLLM,
                 k: int = config.RAG_TOP_K) -> None:
        """Wire a retriever and a generator together.

        Args:
            index: a built DocumentIndex (Week 4). Anything with a compatible
                   `search(query, k) -> list[Passage]` works, which is why the
                   tests can pass a fake.
            llm:   a ready LocalLLM (Week 5). Anything with a compatible
                   `generate` / `stream` works.
            k:     how many passages to retrieve per question.

        Store index, llm, and k on self.
        """

        self.index = index
        self.k = k
        self.llm = llm

    def build_prompt(self, question: str, passages: list[Passage]) -> str:
        """Assemble the user prompt from the question and retrieved passages.

        Returns:
            One string: the numbered context passages, each labelled with its
            source filename, followed by the question. This is sent as the user
            message alongside SYSTEM_PROMPT.

        Behavior:
            - Include each passage's `source` so the model can cite it.
            - Number the passages so the model can refer to them.
            - If `passages` is empty, still return a valid prompt (the model
              should then answer that it does not know).
        """

        lines = ["Context passages:"]

        for i, p in enumerate(passages, 1):
            lines.append(f"[{i}] (source: {p.source}) {p.text}")

        lines.append(f"\nQuestion: {question}")

        return "\n".join(lines)

    def answer(self, question: str) -> RAGAnswer:
        """Answer a question against the indexed documents.

        Behavior (exactly these steps):
            1. passages = self.index.search(question, self.k)
            2. prompt   = self.build_prompt(question, passages)
            3. text     = self.llm.generate(prompt, system=self.SYSTEM_PROMPT)
            4. return RAGAnswer(answer=text, sources=passages, prompt=prompt)
        """

        passages = self.index.search(question, self.k)
        prompt   = self.build_prompt(question, passages)
        text     = self.llm.generate(prompt, system=self.SYSTEM_PROMPT)
        return RAGAnswer(answer=text, sources=passages, prompt=prompt)

    def stream_answer(self, question: str) -> Iterator[str]:
        """Same as answer() but stream the generated text in chunks.

        Retrieve, build the prompt, then `yield from` self.llm.stream(...).
        """

        passages = self.index.search(question, self.k)
        prompt = self.build_prompt(question, passages)
        
        yield from self.llm.stream(prompt, system=self.SYSTEM_PROMPT)