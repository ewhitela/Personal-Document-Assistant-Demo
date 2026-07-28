"""Week 5: Local LLM inference via ollama.

You implement every method marked `raise NotImplementedError`. Keep signatures
stable: Week 6 calls `generate` and `stream`.

Before you start:
    1. Install ollama (https://ollama.com) and start it.
    2. Pull a model: `ollama pull llama3.1:8b`
    3. `pip install ollama`

Libraries you will import when you implement this:
    import ollama        # ollama.Client(host=...).chat(...) / .generate(...) / .list()

Run `python -m tests.test_week5_llm` after implementing (it skips if ollama is
not running, so it is safe to run anytime).
"""

from __future__ import annotations

from collections.abc import Iterator

import ollama

from . import config


class LocalLLM:
    """A thin wrapper over a local ollama model.

    Week 6 calls `generate` with a grounded prompt. The wrapper hides which
    model and host are used, so the rest of the system stays decoupled from
    those choices.
    """

    DEFAULT_SYSTEM = (
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
        self,
        model: str = config.LLM_MODEL,
        host: str = config.LLM_HOST,
        temperature: float = config.LLM_TEMPERATURE,
    ) -> None:
        """Store config and create the ollama client. Do NOT block or download here.

        Hint: `self.client = ollama.Client(host=host)` and keep model/temperature
        on self for later calls.
        """

        self.client = ollama.Client(host=host)

        self.model = model
        self.temperature = temperature
        self.host = host

    def is_ready(self) -> bool:
        """Return True only if the server is reachable AND `self.model` is pulled.

        Behavior:
            - Ask the server for its local models (client.list()).
            - Return True if self.model is among them, else False.
            - On any connection error, return False. Never raise: callers use
              this to fail gracefully.
        """

        try:
            response = self.client.list()  # fetch local models
            models = [m.model for m in response.models]
            return self.model in models
        except Exception:
            return False

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Single-shot completion.

        Args:
            prompt: the user content (in this project, the grounded RAG prompt).
            system: optional system prompt; fall back to DEFAULT_SYSTEM if None.

        Returns:
            The model's reply as plain text.

        Hint: build messages = [{"role": "system", ...}, {"role": "user", ...}]
        and reuse self.chat, or call client.chat and read
        response["message"]["content"].
        """
        messages = [
            {"role": "system", "content": system or self.DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        return self.chat(messages)

    def chat(self, messages: list[dict]) -> str:
        """Multi-message chat.

        Args:
            messages: list of {"role": "system"|"user"|"assistant", "content": str}.

        Returns:
            The assistant's reply text.
        """

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": 220},
            keep_alive=-1,
        )

        return response.message.content

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield the answer in chunks as it is generated.

        Yields:
            Successive text fragments. Concatenating all of them gives the full
            answer. Use the streaming form of the ollama call (stream=True) and
            yield each chunk's text.
        """

        messages = [
            {"role": "system", "content": system or self.DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        for chunk in self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": 220},
            stream=True,
            keep_alive=-1,
        ):
            yield chunk.message.content
