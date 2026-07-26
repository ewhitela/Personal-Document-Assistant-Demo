

"""Week 10: Integration. The orchestrator that runs one turn of the assistant.

Assistant ties every module you built into a single pipeline:

    record audio -> Transcriber (Week 7) -> RAGPipeline (Weeks 4-6) -> Speaker (Week 8)

with an optional VisionModel (Week 9) side path. It depends only on the public
methods of those modules, so you can hand it real modules in the service and fake
ones in the test.

The __init__ (which just stores the modules) is provided. You implement the
methods marked `raise NotImplementedError`: they are the wiring, which is the
whole point of this week.

Run `python tests/test_week10_assistant.py` after implementing. It uses fakes, so
it runs offline with no models.
"""
from __future__ import annotations

from .types import RAGAnswer


class Assistant:
    """Owns one turn of the voice assistant, composed from the week modules."""

    def __init__(self, transcriber, rag, speaker, vision=None) -> None:
        """Store the modules. Dependency injection: pass real modules in the
        service, fakes in the test. `vision` is optional (Week 9).
        """
        self.transcriber = transcriber   # Week 7: has transcribe(path) -> str
        self.rag = rag                   # Week 6: has answer(question) -> RAGAnswer
        self.speaker = speaker           # Week 8: has synthesize(text, path) / say(text)
        self.vision = vision             # Week 9: has ask(image, question) -> str, or None

    def ready(self) -> dict:
        """Return a readiness map for the modules that have is_ready.

        Behavior:
            Build a dict like {"stt": ..., "llm": ..., "tts": ..., "vision": ...}
            by calling is_ready on the transcriber, the rag's llm, the speaker,
            and (if present) the vision model. The service exposes this at /health.
        """
        return {
            "stt": self.transcriber.is_ready() if self.transcriber is not None else False,
            "llm": self.rag.llm.is_ready(),
            "tts": self.speaker.is_ready() if self.speaker is not None else False,
            "vision": self.vision.is_ready() if self.vision is not None else False,
        }


    def answer_text(self, question: str) -> RAGAnswer:
        """Answer a typed question. Behavior: return self.rag.answer(question)."""
        return self.rag.answer(question)
    
    def answer_spoken(self, audio_path: str, out_wav: str | None = None):
        """Answer a spoken question, end to end.

        Behavior:
            1. question = self.transcriber.transcribe(audio_path)
            2. result   = self.answer_text(question)
            3. if out_wav is given, self.speaker.synthesize(result.answer, out_wav)
               otherwise self.speaker.say(result.answer)
            4. return (question, result, out_wav)

        Returns:
            A tuple of the transcript, the RAGAnswer, and the output wav path
            (or None if played directly).
        """

        question = self.transcriber.transcribe(audio_path)
        result = self.answer_text(question)
 
        if out_wav is not None:
            self.speaker.synthesize(result.answer, out_wav)
        else:
            self.speaker.say(result.answer)
 
        return question, result, out_wav

    def speak(self, text: str, out_wav: str | None = None):
        """Speak arbitrary text. synthesize to out_wav if given, else say it."""

        if out_wav is not None:
            self.speaker.synthesize(text, out_wav)
        else:
            self.speaker.say(text)

    def answer_about_image(self, image_path: str, question: str) -> str:
        """Answer a question about an image via the optional vision module.

        Behavior: if self.vision is None, raise a clear error (vision was not
        configured); otherwise return self.vision.ask(image_path, question).
        """

        if self.vision is None:
            raise RuntimeError("Vision module not configured for this assistant")

        return self.vision.ask(image_path, question)

