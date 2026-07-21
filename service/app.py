"""Week 10: FastAPI service that orchestrates every module.

One process owns all the models (loaded once at startup) and exposes them over
HTTP. The Streamlit UI in ui/streamlit_app.py is a thin client of this API.

Run:
    uvicorn service.app:app --host 127.0.0.1 --port 8080

Endpoints:
    GET    /health                 readiness of each component + chunk count
    GET    /documents              filenames currently in the docs dir
    POST   /documents              upload + index one or more files
    DELETE /documents/{filename}   remove one file (reset + re-index the rest)
    DELETE /documents              clear everything
    POST   /ask                    text question -> grounded answer + timings
    POST   /voice/ask              audio question -> transcript, answer, timings,
                                   optional base64 WAV of the spoken answer
    POST   /speak                  text -> WAV bytes

Every answer endpoint returns a per-stage timing breakdown, which feeds the
Week 12 latency deliverable directly.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger("pdva.service")

DOCS_DIR = Path(os.environ.get("PDVA_DOCS_DIR", "pdva_docs")).resolve()
SUPPORTED_DOCS = {".txt", ".md", ".pdf"}
SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


# Component container. Real components are built in build_components();

@dataclass
class Components:
    index: object
    llm: object
    pipeline: object
    transcriber: object | None = None
    speaker: object | None = None


def build_components() -> Components:
    """Load every model once. Import pdva lazily so this module stays cheap."""
    from pdva import DocumentIndex, LocalLLM, RAGPipeline, Speaker, Transcriber

    index = DocumentIndex()
    llm = LocalLLM()
    pipeline = RAGPipeline(index, llm)

    transcriber = None
    try:
        t = Transcriber()
        if t.is_ready():
            transcriber = t
    except Exception:
        logger.exception("Transcriber failed to load; /voice/ask disabled")

    speaker = None
    
    try:
        s = Speaker()
        if s.is_ready():
            speaker = s
    except Exception:
        logger.exception("Speaker failed to load; /speak disabled")

    return Components(index=index, llm=llm, pipeline=pipeline,
                      transcriber=transcriber, speaker=speaker)


def reindex_docs_dir(comp: Components) -> int:
    """Reset the index and re-add every supported file in DOCS_DIR.

    DocumentIndex has no per-document delete, so reset + re-add is the one
    correct rebuild path. Upserts are deterministic (make_chunk_id), so this is
    also safe to call when nothing changed.
    """
    comp.index.reset()

    paths = sorted(str(p) for p in DOCS_DIR.iterdir()
                   if p.suffix.lower() in SUPPORTED_DOCS)
    
    return comp.index.add_documents(paths) if paths else 0



# Schemas

class AskRequest(BaseModel):
    question: str
    speak: bool = False

class SpeakRequest(BaseModel):
    text: str

def _passage_dict(p) -> dict:
    return {"source": p.source, "score": round(p.score, 4),
            "chunk_id": p.chunk_id, "text": p.text}

def create_app(components: Components | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

        if components is not None:
            app.state.comp = components
        else:
            t0 = time.perf_counter()
            app.state.comp = build_components()
            logger.info("components loaded in %.1fs", time.perf_counter() - t0)
        yield

    app = FastAPI(title="PDVA service", lifespan=lifespan)

    def comp() -> Components:
        return app.state.comp

    def answer_with_timings(question: str) -> tuple[dict, dict]:
        c = comp()
        t0 = time.perf_counter()
        passages = c.index.search(question, c.pipeline.k)

        t1 = time.perf_counter()
        prompt = c.pipeline.build_prompt(question, passages)
        text = c.llm.generate(prompt, system=c.pipeline.SYSTEM_PROMPT)

        t2 = time.perf_counter()

        timings = {"retrieve_s": round(t1 - t0, 3),
                   "generate_s": round(t2 - t1, 3)}
        
        body = {"answer": text,
                "sources": [_passage_dict(p) for p in passages]}
        
        return body, timings

    def synthesize_b64(text: str) -> tuple[str, float]:
        c = comp()

        if c.speaker is None:
            raise HTTPException(503, "TTS not available (Piper voice not loaded)")
        
        t0 = time.perf_counter()
        fd, path = tempfile.mkstemp(suffix=".wav")

        os.close(fd)

        try:
            c.speaker.synthesize(text, path)
            data = Path(path).read_bytes()
        finally:
            os.remove(path)

        return base64.b64encode(data).decode("ascii"), round(time.perf_counter() - t0, 3)

    @app.get("/health")
    def health():
        c = comp()

        return {
            "llm_ready": bool(c.llm.is_ready()),
            "stt_ready": c.transcriber is not None,
            "tts_ready": c.speaker is not None,
            "indexed_chunks": c.index.count(),
        }

    @app.get("/documents")
    def list_documents():
        files = sorted(p.name for p in DOCS_DIR.iterdir()
                       if p.suffix.lower() in SUPPORTED_DOCS)
        
        return {"documents": files, "chunks": comp().index.count()}

    @app.post("/documents")
    def upload_documents(files: list[UploadFile] = File(...)):
        saved = []

        for f in files:
            name = Path(f.filename or "").name
            
            if not name or Path(name).suffix.lower() not in SUPPORTED_DOCS:
                raise HTTPException(400, f"Unsupported file type: {f.filename!r} "
                                         f"(supported: {sorted(SUPPORTED_DOCS)})")
            
            dest = DOCS_DIR / name
            
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            
            saved.append(str(dest))
       
        t0 = time.perf_counter()
        chunks = comp().index.add_documents(saved)

        return {"added": [Path(p).name for p in saved],
                "chunks_added": chunks,
                "index_s": round(time.perf_counter() - t0, 3)}

    @app.delete("/documents/{filename}")
    def delete_document(filename: str):
        target = DOCS_DIR / Path(filename).name

        if not target.exists():
            raise HTTPException(404, f"Not indexed: {filename}")
        
        target.unlink()
        chunks = reindex_docs_dir(comp())

        return {"removed": target.name, "chunks_remaining": chunks}

    @app.delete("/documents")
    def clear_documents():
        for p in DOCS_DIR.iterdir():
            if p.suffix.lower() in SUPPORTED_DOCS:
                p.unlink()

        comp().index.reset()

        return {"chunks_remaining": 0}

    @app.post("/ask")
    def ask(req: AskRequest):
        if not req.question.strip():
            raise HTTPException(400, "Empty question")
        
        body, timings = answer_with_timings(req.question)

        if req.speak:
            body["audio_b64"], timings["tts_s"] = synthesize_b64(body["answer"])

        timings["total_s"] = round(sum(v for v in timings.values()), 3)
        body["timings"] = timings

        return body

    @app.post("/voice/ask")
    def voice_ask(audio: UploadFile = File(...), speak: bool = False):
        c = comp()

        if c.transcriber is None:
            raise HTTPException(503, "STT not available (whisper model not loaded)")
        
        suffix = Path(audio.filename or "q.wav").suffix.lower() or ".wav"

        if suffix not in SUPPORTED_AUDIO:
            raise HTTPException(400, f"Unsupported audio type: {suffix}")
        
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        try:
            with open(path, "wb") as out:
                shutil.copyfileobj(audio.file, out)
            t0 = time.perf_counter()
            transcript = c.transcriber.transcribe(path)
            stt_s = round(time.perf_counter() - t0, 3)
        finally:
            os.remove(path)

        if not transcript.strip():
            return {"transcript": "", "answer": "", "sources": [],
                    "timings": {"stt_s": stt_s, "total_s": stt_s}}
        
        body, timings = answer_with_timings(transcript)
        timings = {"stt_s": stt_s, **timings}

        if speak:
            body["audio_b64"], timings["tts_s"] = synthesize_b64(body["answer"])

        timings["total_s"] = round(sum(v for v in timings.values()), 3)
        body["transcript"] = transcript
        body["timings"] = timings

        return body

    # -- speak -------------------------------------------------------------
    @app.post("/speak")
    def speak(req: SpeakRequest):
        if not req.text.strip():
            raise HTTPException(400, "Empty text")
        
        b64, _ = synthesize_b64(req.text)

        return Response(content=base64.b64decode(b64), media_type="audio/wav")

    return app


app = create_app()
