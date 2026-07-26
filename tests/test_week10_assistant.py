

"""Sanity tests for the Week 10 Assistant orchestrator.

    python tests/test_week10_assistant.py

These use fake modules, so they run offline with no models. They check that the
Assistant wires the pipeline correctly: a transcript flows into the RAG answer,
the answer flows into the speaker, the readiness map aggregates the modules, and
the optional vision path is routed.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.assistant import Assistant
from pdva.types import Passage, RAGAnswer


class FakeTranscriber:
    def is_ready(self):
        return True
    def transcribe(self, audio_path):
        return "what is the refund window?"


class FakeLLM:
    def is_ready(self):
        return True
    def generate(self, prompt, system=None):
        return "The refund window is 30 days. [refunds.txt]"


class FakeRAG:
    """Mimics RAGPipeline: holds an llm and answers questions."""
    def __init__(self):
        self.llm = FakeLLM()
        self.last_question = None
    def answer(self, question):
        self.last_question = question
        src = [Passage("The refund window is 30 days.", "refunds.txt", 0.9, "refunds.txt::0")]
        return RAGAnswer(answer=self.llm.generate("prompt"), sources=src, prompt="prompt")


class FakeSpeaker:
    def __init__(self):
        self.spoken = []
    def is_ready(self):
        return True
    def synthesize(self, text, out_path):
        self.spoken.append(text)
        with open(out_path, "wb") as f:
            f.write(b"RIFF....WAVE")
        return out_path
    def say(self, text):
        self.spoken.append(text)


class FakeVision:
    def is_ready(self):
        return True
    def ask(self, image_path, question):
        return f"description of {os.path.basename(image_path)}"


def _bot():
    return Assistant(FakeTranscriber(), FakeRAG(), FakeSpeaker(), vision=FakeVision())


def test_answer_text_routes_to_rag():
    bot = _bot()
    res = bot.answer_text("how long for a refund?")
    assert isinstance(res, RAGAnswer) and "30 days" in res.answer
    assert bot.rag.last_question == "how long for a refund?", "should pass the question to rag.answer"


def test_answer_spoken_chains_stt_rag_tts():
    bot = _bot()
    out = os.path.join(tempfile.mkdtemp(), "reply.wav")
    q, res, wav = bot.answer_spoken("question.wav", out_wav=out)
    assert q == "what is the refund window?", "transcript should come from the transcriber"
    assert isinstance(res, RAGAnswer) and "30 days" in res.answer, "rag should answer the transcript"
    assert res.answer in bot.speaker.spoken, "the answer should be sent to the speaker"
    assert os.path.exists(wav), "a wav should be written when out_wav is given"


def test_ready_aggregates_modules():
    bot = _bot()
    status = bot.ready()
    assert status.get("stt") is True and status.get("llm") is True and status.get("tts") is True
    assert status.get("vision") is True, "vision status should be present when a vision module is set"


def test_answer_about_image_routes_to_vision():
    bot = _bot()
    out = bot.answer_about_image("/tmp/diagram.png", "what is this?")
    assert out == "description of diagram.png", "should route to vision.ask"


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

