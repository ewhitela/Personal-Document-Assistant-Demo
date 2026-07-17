

"""Week 9 (optional): Visual input with a swappable backend.

Build this only if the pace allows. It answers questions about an image, and it
is designed for two deployment targets behind one API:

    - LOCAL  : a small model (moondream) served by ollama on the same 11 GB GPU
               as the text LLM. Limited, but enough to demo the whole system on
               one box.
    - REMOTE : a stronger model on a separate inference server with its own GPU,
               reached over HTTP (a custom service, Triton, an OpenAI-compatible
               endpoint, and so on).

The rest of the system calls VisionModel.ask / describe and never knows which
backend is active. You implement the two backends and the two helper functions
marked `raise NotImplementedError`. The VisionModel facade and its .local() /
.remote() constructors are provided.

Before you start (local backend):
    ollama pull moondream
    pip install ollama
Before you start (remote backend):
    pip install requests    # and stand up your remote vision server

Run `python tests/test_week9_vision.py` after implementing.
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod

from . import config

import ollama
import requests


# ---------------------------------------------------------------------------
# Small pure helpers (offline-testable). Implement these.
# ---------------------------------------------------------------------------
def build_image_message(image_path: str, question: str) -> dict:
    """Build the ollama chat message that carries an image.

    Returns:
        {"role": "user", "content": question, "images": [image_path]}
    ollama accepts a file path or raw bytes in the images list.
    """

    return {"role": "user", "content": question, "images": [image_path]}

def encode_image_b64(image_path: str) -> str:
    """Read an image file and return its base64-encoded contents as ASCII text.

    Used by the remote backend, which must send image bytes over HTTP.
    Behavior: read the file in binary, base64-encode, decode to an ASCII str.
    """
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Backend interface. Both concrete backends satisfy this.
# ---------------------------------------------------------------------------
class VisionBackend(ABC):
    """Common interface so local and remote inference are interchangeable."""

    @abstractmethod
    def is_ready(self) -> bool:
        """True if this backend can serve a request right now. Never raises."""

    @abstractmethod
    def ask(self, image_path: str, question: str) -> str:
        """Answer a question about the image and return the reply text."""


class OllamaVisionBackend(VisionBackend):
    """Local backend: a small vision model served by ollama on this machine."""

    def __init__(self, model: str = config.VISION_MODEL_LOCAL,
                 host: str = config.LLM_HOST,
                 temperature: float = config.LLM_TEMPERATURE) -> None:
        """Store config and create self.client = ollama.Client(host=host).
        Do not block or download here.
        """

        self.client = ollama.Client(host=host)
        self.model = model
        self.host = host
        self.temperature = temperature

    def is_ready(self) -> bool:
        try:
            resp = self.client.list()
            models = resp.get("models", resp.models if hasattr(resp, "models") else [])
            names = []
            for m in models:
                if isinstance(m, dict):
                    names.append(m.get("name") or m.get("model", ""))
                else:
                    names.append(getattr(m, "model", "") or getattr(m, "name", ""))
            return any(self.model in n for n in names)
        except Exception:
            return False

    def ask(self, image_path: str, question: str) -> str:
        """Send one image message to the model and return the reply text.

        Hint: msg = build_image_message(image_path, question); then
        self.client.chat(model=self.model, messages=[msg],
                         options={"temperature": self.temperature}), and read
        response["message"]["content"].
        """

        msg = build_image_message(image_path, question); 

        response = self.client.chat(
            model=self.model,
            messages=[msg],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]


class RemoteVisionBackend(VisionBackend):
    """Remote backend: a stronger vision model on a separate HTTP server.

    The exact request and response schema depends on your server (a custom
    FastAPI service, Triton via KServe, an OpenAI-compatible endpoint, ...).
    build_payload and parse_response are the two seams to override for your
    server; a simple default JSON contract is provided.
    """

    def __init__(self, endpoint: str = config.VISION_REMOTE_URL,
                 model: str = config.VISION_MODEL_REMOTE,
                 timeout: int = config.VISION_REMOTE_TIMEOUT,
                 api_key: str | None = None) -> None:
        """Store the endpoint, model, timeout, and optional api_key on self.
        This is plain configuration; nothing to contact yet.
        """

        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def build_payload(self, question: str, image_b64: str) -> dict:
        """Return the JSON body to POST. Override to match your server.

        Default contract:
            {"model": <name>, "question": <text>, "image_b64": <base64 image>}
        """
        return {"model": self.model, "question": question, "image_b64": image_b64}

    def parse_response(self, data: dict) -> str:
        """Extract the answer text from the server's JSON. Override to match.

        Default: data["answer"].
        """
        return data["answer"]

    def is_ready(self) -> bool:
        """True if the remote server is reachable. Never raises.

        Behavior: make a lightweight request (a health check, or a HEAD/GET on
        the endpoint) inside try/except and return False on any error.
        """
        
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = requests.get(self.endpoint, headers=headers, timeout=self.timeout)
            return r.status_code < 500
        except Exception:
            return False


    def ask(self, image_path: str, question: str) -> str:
        """Answer a question about the image via the remote server.

        Behavior:
            b64     = encode_image_b64(image_path)
            payload = self.build_payload(question, b64)
            POST payload as JSON to self.endpoint (add an Authorization header if
            self.api_key is set), honour self.timeout, raise for HTTP errors,
            then return self.parse_response(response.json()).
        """

        b64 = encode_image_b64(image_path)
        payload = self.build_payload(question, b64)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return self.parse_response(r.json())

# ---------------------------------------------------------------------------
# Facade the rest of the system uses. Provided, not a stub.
# ---------------------------------------------------------------------------
class VisionModel:
    """One vision API over a chosen backend.

    Default is the local ollama backend. Use the constructors below to be
    explicit, or pass any VisionBackend (including a fake in tests).
    """

    def __init__(self, backend: VisionBackend | None = None) -> None:
        self.backend: VisionBackend = backend if backend is not None else OllamaVisionBackend()

    @classmethod
    def local(cls, model: str = config.VISION_MODEL_LOCAL,
              host: str = config.LLM_HOST) -> "VisionModel":
        """A VisionModel backed by a local ollama vision model."""
        return cls(OllamaVisionBackend(model=model, host=host))

    @classmethod
    def remote(cls, endpoint: str = config.VISION_REMOTE_URL,
               model: str = config.VISION_MODEL_REMOTE,
               api_key: str | None = None) -> "VisionModel":
        """A VisionModel backed by a remote HTTP vision server."""
        return cls(RemoteVisionBackend(endpoint=endpoint, model=model, api_key=api_key))

    def is_ready(self) -> bool:
        return self.backend.is_ready()

    def ask(self, image_path: str, question: str) -> str:
        return self.backend.ask(image_path, question)

    def describe(self, image_path: str) -> str:
        return self.ask(image_path, "Describe this image in detail.")

