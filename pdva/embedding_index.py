"""Week 4: Document embedding and retrieval.

You implement every function and method marked `raise NotImplementedError`.
Do NOT change the signatures: Week 6 (the RAG pipeline) depends on them exactly.

Libraries you will import when you implement this:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from pypdf import PdfReader        # or: import fitz  (PyMuPDF)

Run `python -m tests.test_week4_index` after implementing to check your work.
"""

from __future__ import annotations
from .types import Passage
from . import config
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os

# ---------------------------------------------------------------------------
# Module-level helpers. Implement these first; the class uses them.
# ---------------------------------------------------------------------------
def extract_text(path: str) -> str:

    """
    Read a document and return its plain text.

    Args:
        path: path to a .txt, .md, or .pdf file.

    Returns:
        The full text content as one string.

    Behavior:
        - .txt / .md : read as UTF-8 text.
        - .pdf       : extract every page's text and join with newlines.
        - any other extension: raise ValueError.
    """

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    elif ext == ".md" or ext == ".txt":
        with open(path, encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str,
               chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text:       the document text.
        chunk_size: target chunk length. Counting words is a fine starting proxy.
        overlap:    how many words two neighbouring chunks share.

    Returns:
        A list of non-empty chunks, in original order.

    Behavior:
        - Neighbouring chunks overlap by `overlap` words so a sentence split
          across a boundary is not lost.
        - The last chunk may be shorter than chunk_size.
        - Never return empty strings.
        - Require 0 <= overlap < chunk_size (raise ValueError otherwise).
        - Keep chunk_size within the embedding model's max input. all-MiniLM-L6-v2
          caps at 256 tokens (~200 words); longer chunks are silently truncated.
    """

    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

    


def make_chunk_id(source: str, index: int) -> str:
    """Return a deterministic id for a chunk.

    The same (source, index) must always map to the same id, so that
    re-indexing a document updates its chunks instead of duplicating them.
    Example shape: f"{source}::chunk_{index}".
    """

    return f"{source}::chunk_{index}"

    


# ---------------------------------------------------------------------------
# The index itself.
# ---------------------------------------------------------------------------
class DocumentIndex:
    """A persistent, searchable index of document chunks.

    This is the assistant's memory. Week 6 calls `search` to fetch the passages
    it feeds to the LLM. The index must survive a process restart.
    """

    def __init__(self,
                 persist_dir: str = config.PERSIST_DIR,
                 embedding_model: str = config.EMBEDDING_MODEL,
                 collection_name: str = config.COLLECTION_NAME) -> None:
        """Open or create a persistent vector store.

        Behavior:
            - Create a Chroma PersistentClient at `persist_dir`.
            - get_or_create a collection named `collection_name`. Create it with
              cosine distance, metadata={"hnsw:space": "cosine"}, so that
              score = 1 - distance is a clean 0..1 similarity in `search`.
            - Decide how chunks get embedded. Two valid approaches:
                (a) load a SentenceTransformer(embedding_model) and embed text
                    yourself, passing vectors to Chroma, or
                (b) hand Chroma a SentenceTransformerEmbeddingFunction so it
                    embeds for you. Pick one and be consistent.
            - Store enough state on self to implement the methods below.
        """

        self.client = chromadb.PersistentClient(path=persist_dir)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name

        self.model = SentenceTransformer(embedding_model)

    def add_documents(self, paths: list[str]) -> int:
        """Index one or more documents.

        Args:
            paths: file paths to ingest.

        Returns:
            The number of chunks added or updated.

        Behavior:
            For each path:
                text   = extract_text(path)
                chunks = chunk_text(text)
                ids    = [make_chunk_id(filename, i) for i, _ in enumerate(chunks)]
                upsert chunks into the collection with metadata
                    {"source": filename, "chunk": i}.
            Use upsert (not add) so re-indexing the same file does not duplicate.
        """
        total = 0

        for path in paths:
            filename = os.path.basename(path)

            text   = extract_text(path)
            chunks = chunk_text(text)

            if not chunks:
                continue

            ids    = [make_chunk_id(filename, i) for i, _ in enumerate(chunks)]
            metas  = [{"source": filename, "chunk": i} for i in range(len(chunks))]
            embeddings = self.model.encode(chunks).tolist()

            self.collection.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
            total += len(chunks)

        return total

    def search(self, query: str, k: int = config.RAG_TOP_K) -> list[Passage]:
        """Return the k most relevant chunks for a query.

        Args:
            query: a natural-language question.
            k:     number of passages to return.

        Returns:
            Up to k Passage objects, most relevant first. Convert Chroma's
            distance into a similarity score where higher = more relevant
            (e.g. score = 1.0 - distance) and put it in Passage.score.
        """

        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )

        passages = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            passages.append(Passage(
                text=doc,
                source=meta["source"],
                score=1.0 - dist,
                chunk_id=make_chunk_id(meta["source"], meta["chunk"]),
                metadata=meta,
            ))

        return passages

    def count(self) -> int:
        """Return how many chunks are currently stored."""
        
        return self.collection.count()
    
    def reset(self) -> None:
        """Remove all chunks from the collection (used by tests and re-builds)."""

        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(self, texts: list[str], sources: list[str]) -> int:
        """Index raw text strings directly, without needing files on disk.
    
        For each (text, source): chunk_text -> embed -> upsert with
        make_chunk_id(source, i) and metadata {"source": source, "chunk": i}.
        add_documents can call this after extract_text.
        """

        total = 0

        for text, source in zip(texts, sources):
            chunks = chunk_text(text)

            if not chunks:
                continue

            ids   = [make_chunk_id(source, i) for i, _ in enumerate(chunks)]
            metas = [{"source": source, "chunk": i} for i in range(len(chunks))]
            embeddings = self.model.encode(chunks).tolist()

            self.collection.upsert(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
            total += len(chunks)

        return total