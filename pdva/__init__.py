"""Personal Document Voice Assistant (PDVA) starter package.

You build three modules across Weeks 4 to 6:

    embedding_index.DocumentIndex   Week 4   document embedding and retrieval
    llm.LocalLLM                    Week 5   local LLM inference via ollama
    rag.RAGPipeline                 Week 6   retrieval-augmented generation

The signatures are fixed so the modules plug into each other (and into the
Week 10 service) without changes. Fill in the bodies marked NotImplementedError.
"""
from .types import Passage, RAGAnswer
from .embedding_index import DocumentIndex
from .llm import LocalLLM
from .rag import RAGPipeline

__all__ = ["Passage", "RAGAnswer", "DocumentIndex", "LocalLLM", "RAGPipeline"]
