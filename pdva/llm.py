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

from typing import Iterator

from . import config

import ollama


class LocalLLM:
    """A thin wrapper over a local ollama model.

    Week 6 calls `generate` with a grounded prompt. The wrapper hides which
    model and host are used, so the rest of the system stays decoupled from
    those choices.
    """

    DEFAULT_SYSTEM = (
        "You are a precise assistant for a personal document collection. "
        "Answer the question using ONLY the context passages provided, directly "
        "and completely in 2-4 sentences, including the key specifics from the "
        "context. If the context does not contain the answer, reply exactly: "
        "\"I don't know based on your documents.\""
    )

    def __init__(self,
                 model: str = config.LLM_MODEL,
                 host: str = config.LLM_HOST,
                 temperature: float = config.LLM_TEMPERATURE) -> None:
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
            response = self.client.list() # fetch local models
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
        messages = [{'role': 'system', 'content': system or self.DEFAULT_SYSTEM},
                    {'role': 'user', 'content': prompt}]

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
            options={"temperature": self.temperature, "num_predict": 220}
        )

        return response.message.content
    
    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield the answer in chunks as it is generated.

        Yields:
            Successive text fragments. Concatenating all of them gives the full
            answer. Use the streaming form of the ollama call (stream=True) and
            yield each chunk's text.
        """

        messages = [{'role': 'system', 'content': system or self.DEFAULT_SYSTEM}, {'role': 'user', 'content': prompt}]
        
        for chunk in self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": 220},
            stream=True
        ):
            yield chunk.message.content