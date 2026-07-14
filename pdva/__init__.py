

"""Personal Document Voice Assistant (PDVA) starter package.

You build these modules across Weeks 4 to 9:

    embedding_index.DocumentIndex   Week 4   document embedding and retrieval
    llm.LocalLLM                    Week 5   local LLM inference via ollama
    rag.RAGPipeline                 Week 6   retrieval-augmented generation
    transcriber.Transcriber         Week 7   speech-to-text via faster-whisper
    tts.Speaker                     Week 8   text-to-speech via Piper
    vision.VisionModel              Week 9   visual input via ollama (optional)

The signatures are fixed so the modules plug into each other and into the
Week 10 service without changes. Fill in the bodies marked NotImplementedError.
"""
from .types import Passage, RAGAnswer, TranscriptSegment
from .embedding_index import DocumentIndex
from .llm import LocalLLM
from .rag import RAGPipeline
from .transcriber import Transcriber
from .tts import Speaker
from .vision import VisionModel, VisionBackend, OllamaVisionBackend, RemoteVisionBackend

__all__ = [
    "Passage", "RAGAnswer", "TranscriptSegment",
    "DocumentIndex", "LocalLLM", "RAGPipeline",
    "Transcriber", "Speaker",
    "VisionModel", "VisionBackend", "OllamaVisionBackend", "RemoteVisionBackend",
]

