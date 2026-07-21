

"""One place for the settings every module shares.

Change a default here and the whole pipeline agrees. Beginners: you do not need
to touch this to get started; the defaults are sensible.
"""
from __future__ import annotations

from pathlib import Path

# Week 4: where the vector index lives on disk, and what the collection is called.
PERSIST_DIR = str((Path.cwd() / "pdva_store").resolve())
COLLECTION_NAME = "documents"

# Week 4: embedding model and chunking. Swap EMBEDDING_MODEL to compare quality.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # try "BAAI/bge-small-en-v1.5" later
# Chunk length is in words (a beginner-friendly proxy for tokens). 200 words is
# ~260 tokens, which stays close to all-MiniLM-L6-v2's 256-token input cap so
# chunks are not silently truncated. If you switch to a 512-token model you can
# raise this. The precise approach is to chunk by tokens with the model tokenizer.
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30                     # words shared between neighbouring chunks (~15%)

# Week 5: which local model to call, where the ollama server is, how random.
LLM_MODEL = "llama3.2:3b"              # must be pulled first: `ollama pull llama3.1:8b`
LLM_HOST = "http://localhost:11434"
LLM_TEMPERATURE = 0.2                  # low = more deterministic, good for grounded Q&A

# Week 6: how many passages to retrieve per question.
RAG_TOP_K = 3

# Week 7: speech-to-text (faster-whisper).
WHISPER_MODEL = "base.en"     # tiny.en / base.en / small.en. ".en" is English-only and faster.
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

# Week 8: text-to-speech (Piper).
# Download the voice first: python -m piper.download_voices en_US-lessac-medium
PIPER_VOICE = "en_US-lessac-medium.onnx"
PIPER_USE_CUDA = False        # True needs onnxruntime-gpu installed.

# Week 9 (optional): vision. One API, two deployment targets.
# Local backend: a small model that co-resides with the text LLM on one 11 GB GPU,
# enough to demo the whole system end to end.
VISION_MODEL_LOCAL = "moondream"          # ~2 GB, fits alongside the Week 5 text model
# Remote backend: a stronger model on a separate inference server (its own GPU),
# reached over HTTP (a custom service, Triton, an OpenAI-compatible endpoint, ...).
VISION_MODEL_REMOTE = "qwen2.5vl:7b"      # the better model you host elsewhere
VISION_REMOTE_URL = "http://localhost:8000/vision"   # your remote endpoint
VISION_REMOTE_TIMEOUT = 30                # seconds

# OWW Model

OPENWAKEWORD_MODEL = "jarona.onnx"