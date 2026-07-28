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

from collections.abc import Iterator

from . import config
from .embedding_index import DocumentIndex
from .llm import LocalLLM
from .types import Passage, RAGAnswer


class RAGPipeline:
    """Retrieve, then generate a grounded answer."""

    SYSTEM_PROMPT = (
        "You are a precise assistant for a personal document collection: syllabi, "
        "leases, contracts, articles, and similar. Your answer will be read aloud, "
        "so keep it short and plain.\n\n"
        "Use ONLY the context passages provided. Two rules matter most:\n\n"
        "1. Do not assert that something is still in effect, current, or ongoing "
        "unless the passages say so, even if it seems like a reasonable assumption "
        "or is common knowledge. If the passages describe something only as of a "
        "past date, or do not state its current status, say nothing about its "
        "current status. This rule is about currency and status only. It does not "
        "stop you from stating a fact the passages do give, or from drawing the "
        "plain conclusion that fact supports.\n\n"
        "2. When a passage compares two things, assigns different terms to "
        "different parties, or states a relationship between two items (who owes "
        "what, which is greater, which came first), restate it in the same "
        "direction the passage states it. Check the direction before you write.\n\n"
        "Length: at most 3 sentences, and stop as soon as the question is "
        "answered. One sentence is correct and preferred when one sentence is "
        "enough. Open the answer with the specific figure, name, or date asked "
        "for. For a yes or no question, give the yes or no and then the single "
        "most specific supporting detail from the passage. Do not restate the "
        "question, list your sources, or add a closing summary.\n\n"
        "Refusing: reply with the exact refusal only when no passage contains any "
        "fact bearing on the question. A passage counts as bearing on the question "
        "only if it states a fact that answers, bounds, or partly answers the "
        "specific property or claim being asked about, not merely because it "
        "discusses the same subject. A passage about the same person, place, or "
        "species that does not address the property asked (for example, a passage "
        "about markings when the question asks about venom) does not bear on the "
        "question; treat that as if the context contains nothing on the question. "
        "If a passage does state something that bears on the question but not the "
        "exact claim asked for, give that fact and say plainly what it does and "
        "does not establish. When the context contains "
        "nothing at all on the question, your entire reply must be this exact "
        "sentence, with no additions or rewording: \"I don't know based on your "
        'documents." Do not explain what the context lacks or describe the '
        "passages. If the context answers only part of the question, give that "
        "part and name what is missing in a short clause, not a separate "
        "sentence."
    )

    def __init__(
        self, index: DocumentIndex, llm: LocalLLM, k: int = config.RAG_TOP_K
    ) -> None:
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
        prompt = self.build_prompt(question, passages)
        text = self.llm.generate(prompt, system=self.SYSTEM_PROMPT)
        return RAGAnswer(answer=text, sources=passages, prompt=prompt)

    def stream_answer(self, question: str) -> Iterator[str]:
        """Same as answer() but stream the generated text in chunks.

        Retrieve, build the prompt, then `yield from` self.llm.stream(...).
        """

        passages = self.index.search(question, self.k)
        prompt = self.build_prompt(question, passages)

        yield from self.llm.stream(prompt, system=self.SYSTEM_PROMPT)
