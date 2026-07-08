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
LLM_MODEL = "llama3.1:8b"              # must be pulled first: `ollama pull llama3.1:8b`
LLM_HOST = "http://localhost:11434"
LLM_TEMPERATURE = 0.2                  # low = more deterministic, good for grounded Q&A

# Week 6: how many passages to retrieve per question.
RAG_TOP_K = 4
