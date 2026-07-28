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
    POST   /voice/wake/start       start the background wake-word listener
    POST   /voice/wake/stop        stop it
    GET    /voice/wake/status      current state (idle/listening/recording/answering/error)
    GET    /voice/wake/result      pop the next completed turn, if any

Every answer endpoint returns a per-stage timing breakdown, which feeds the
Week 12 latency deliverable directly.

Wake-word note: the listener opens the mic on the machine running this
service process, not the browser running Streamlit. That's fine for this
single-workstation demo (service + mic + UI are all on the same box) but
would need rethinking if the UI were ever served to a different machine.
"""

from __future__ import annotations

import base64
import logging
import os
import queue as pyqueue
import shutil
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from pdva.assistant import Assistant
from pdva.verifier import VerifiedRAGPipeline, verify

logger = logging.getLogger("pdva.service")

DOCS_DIR = Path(os.environ.get("PDVA_DOCS_DIR", "pdva_docs")).resolve()
SUPPORTED_DOCS = {".txt", ".md", ".pdf"}
SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# Wake-word audio constants (mirrors voice_demo.py).
WAKE_SAMPLE_RATE = 16000
WAKE_OWW_FRAME_SAMPLES = 1280  # openWakeWord requires 80ms int16 frames
WAKE_VAD_FRAME_MS = 30  # webrtcvad only accepts 10/20/30ms
WAKE_VAD_FRAME_SAMPLES = WAKE_SAMPLE_RATE * WAKE_VAD_FRAME_MS // 1000
WAKE_THRESHOLD = 0.5
WAKE_SILENCE_FRAMES_TO_STOP = int(1.5 * 1000 / WAKE_VAD_FRAME_MS)
WAKE_MAX_UTTERANCE_FRAMES = int(15 * 1000 / WAKE_VAD_FRAME_MS)  # 15s hard cap
WAKE_VAD_AGGRESSIVENESS = 1


@dataclass
class Components:
    # `index` stays separate: Assistant has no reference to it (it only holds
    # the rag pipeline, transcriber, speaker, vision), but the /documents
    # endpoints need to add/reset/count chunks directly.
    index: object
    assistant: Assistant


def build_components() -> Components:
    """Load every model once. Import pdva lazily so this module stays cheap."""
    from pdva import (
        DocumentIndex,
        LocalLLM,
        RAGPipeline,
        Speaker,
        Transcriber,
        VisionModel,
    )

    index = DocumentIndex()
    llm = LocalLLM()
    pipeline = VerifiedRAGPipeline(RAGPipeline(index, llm))

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

    vision = None
    try:
        v = VisionModel.local()
        if v.is_ready():
            vision = v
    except Exception:
        logger.exception("Vision failed to load; /vision/ask disabled")

    assistant = Assistant(
        transcriber=transcriber, rag=pipeline, speaker=speaker, vision=vision
    )
    return Components(index=index, assistant=assistant)


def reindex_docs_dir(comp: Components) -> int:
    """Reset the index and re-add every supported file in DOCS_DIR.

    DocumentIndex has no per-document delete, so reset + re-add is the one
    correct rebuild path. Upserts are deterministic (make_chunk_id), so this is
    also safe to call when nothing changed.
    """
    comp.index.reset()

    paths = sorted(
        str(p) for p in DOCS_DIR.iterdir() if p.suffix.lower() in SUPPORTED_DOCS
    )

    return comp.index.add_documents(paths) if paths else 0


def _passage_dict(p) -> dict:
    return {
        "source": p.source,
        "score": round(p.score, 4),
        "chunk_id": p.chunk_id,
        "text": p.text,
    }


# -- Shared answer/speak helpers --------------------------------------------
# Pulled out of create_app (rather than left as closures) so the wake-word
# background thread can call the exact same retrieve/generate/verify path as
# /ask and /voice/ask -- one code path, one set of timings, no drift between
# the three entry points.


def answer_with_timings(c: Components, question: str) -> tuple[dict, dict]:
    """Retrieve + generate, timed as two stages.

    This goes through c.assistant.rag (the RAGPipeline) rather than
    assistant.answer_text(), because answer_text() returns a single
    RAGAnswer with no internal timing split, and retrieve_s/generate_s
    are graded separately in the Week 12 latency breakdown. Both
    c.assistant.rag and c.assistant.rag.llm are the same objects
    answer_text() would use internally -- this is still "going through
    the assistant", just at the stage-level instead of the single
    convenience call.
    """
    rag = c.assistant.rag

    t0 = time.perf_counter()
    passages = c.index.search(question, rag.k)

    t1 = time.perf_counter()
    prompt = rag.build_prompt(question, passages)
    text = rag.llm.generate(prompt, system=rag.SYSTEM_PROMPT)

    t2 = time.perf_counter()
    text, report = verify(text, passages, question)
    t3 = time.perf_counter()

    timings = {
        "retrieve_s": round(t1 - t0, 3),
        "generate_s": round(t2 - t1, 3),
        "verify_s": round(t3 - t2, 3),
    }

    body = {
        "answer": text,
        "sources": [_passage_dict(p) for p in passages],
        "verification": {
            "passed": report.passed,
            "abstained": report.abstained,
            "ungrounded_numbers": report.ungrounded,
            "ungrounded_entities": report.ungrounded_entities,
            "status_stripped": report.status_stripped,
        },
    }

    return body, timings


def synthesize_b64(c: Components, text: str) -> tuple[str, float]:
    speaker = c.assistant.speaker

    if speaker is None:
        raise HTTPException(503, "TTS not available (Piper voice not loaded)")

    t0 = time.perf_counter()
    fd, path = tempfile.mkstemp(suffix=".wav")

    os.close(fd)

    try:
        speaker.synthesize(text, path)
        data = Path(path).read_bytes()
    finally:
        os.remove(path)

    return base64.b64encode(data).decode("ascii"), round(time.perf_counter() - t0, 3)


class WakeListener:
    """Background wake-word -> VAD -> STT -> RAG loop, running in its own thread.

    Same wake word ("jarona"), VAD, and end-pointing logic as voice_demo.py,
    but instead of printing the transcript it feeds it through
    answer_with_timings() / synthesize_b64() -- the same functions /ask and
    /voice/ask use -- and pushes the finished turn onto a queue for the API
    (and the Streamlit poll loop) to pick up.

    One listener per process. Starting while already running is a no-op;
    call stop() and wait for state to reach "idle" before starting again
    with different options (e.g. toggling `speak`).
    """

    def __init__(self, comp: Components, speak: bool = False):
        self.comp = comp
        self.speak = speak
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = "idle"  # idle | listening | recording | answering | error: <msg>
        self.last_score = 0.0  # most recent wake-word confidence, for live debugging
        self.results: pyqueue.Queue[dict] = pyqueue.Queue()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _read_frame(self, q, n_samples: int, leftover):
        """Pulls exactly n_samples int16 samples, buffering across queue reads.

        Returns (frame, new_leftover); frame is None if stop was requested
        before enough samples arrived.
        """
        import numpy as np

        buf = leftover
        while buf.shape[0] < n_samples:
            if self._stop.is_set():
                return None, buf
            try:
                chunk = q.get(timeout=0.5).reshape(-1)
            except pyqueue.Empty:
                continue
            buf = np.concatenate([buf, chunk])
        return buf[:n_samples], buf[n_samples:]

    def _run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            import webrtcvad
            from openwakeword.model import Model as WakeWordModel

            from pdva import config
        except Exception as e:
            self.state = f"error: missing voice dependency ({e})"
            return

        try:
            oww = WakeWordModel(wakeword_models=[config.OPENWAKEWORD_MODEL])
        except Exception as e:
            self.state = f"error: could not load wake-word model ({e})"
            return

        vad = webrtcvad.Vad(WAKE_VAD_AGGRESSIVENESS)
        q: pyqueue.Queue = pyqueue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("wake listener audio status: %s", status)
            q.put(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=WAKE_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=WAKE_VAD_FRAME_SAMPLES,
                callback=callback,
            )
        except Exception as e:
            self.state = f"error: could not open microphone ({e})"
            return

        leftover = np.zeros(0, dtype=np.int16)
        frames: list = []  # only populated if a complete, non-aborted utterance is captured

        try:
            with stream:
                self.state = "listening"

                frame_count = 0

                while not self._stop.is_set():
                    frame, leftover = self._read_frame(
                        q, WAKE_OWW_FRAME_SAMPLES, leftover
                    )
                    if frame is None:
                        break

                    score = max(oww.predict(frame).values())
                    self.last_score = round(float(score), 3)
                    frame_count += 1

                    if frame_count % 12 == 0:
                        logger.info(
                            "wake listener score=%.3f (threshold=%.2f)",
                            self.last_score,
                            WAKE_THRESHOLD,
                        )

                    if score < WAKE_THRESHOLD:
                        continue

                    self.state = "recording"
                    utterance_frames = []
                    silence_run = 0
                    n = 0

                    while n < WAKE_MAX_UTTERANCE_FRAMES and not self._stop.is_set():
                        vframe, leftover = self._read_frame(
                            q, WAKE_VAD_FRAME_SAMPLES, leftover
                        )
                        if vframe is None:
                            break
                        utterance_frames.append(vframe)
                        is_speech = vad.is_speech(vframe.tobytes(), WAKE_SAMPLE_RATE)
                        silence_run = 0 if is_speech else silence_run + 1
                        n += 1
                        if (
                            silence_run >= WAKE_SILENCE_FRAMES_TO_STOP
                            and n > silence_run
                        ):
                            break

                    # A stop request mid-question means abort cleanly -- no
                    # answering pass on a truncated utterance. Only hand off
                    # to _handle_utterance if the listener wasn't stopped
                    # while still recording.
                    if not self._stop.is_set():
                        frames = utterance_frames

                    # One utterance (or one aborted attempt) per activation --
                    # the mic closes below either way; re-arming needs an
                    # explicit start() (the record button).
                    break

            if frames:
                self.state = "answering"
                self._handle_utterance(np.concatenate(frames))

        except Exception as e:
            logger.exception("wake listener crashed")
            self.state = f"error: {e}"
            return

        self.state = "idle"

    def _drain_for(self, q, seconds: float) -> None:
        """Discard incoming audio for `seconds`.

        Used after speaking an answer so the TTS played back through this
        machine's speakers isn't recaptured by the mic as a new wake-word
        trigger or utterance.
        """
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and not self._stop.is_set():
            try:
                q.get(timeout=0.1)
            except pyqueue.Empty:
                continue

    def _handle_utterance(self, audio_i16) -> None:
        transcriber = self.comp.assistant.transcriber

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)  # int16
                w.setframerate(WAKE_SAMPLE_RATE)
                w.writeframes(audio_i16.tobytes())

            t0 = time.perf_counter()
            transcript = transcriber.transcribe(path)
            stt_s = round(time.perf_counter() - t0, 3)
        finally:
            os.remove(path)

        if not transcript.strip():
            self.results.put(
                {
                    "transcript": "",
                    "answer": "",
                    "sources": [],
                    "timings": {"stt_s": stt_s, "total_s": stt_s},
                }
            )
            return

        body, timings = answer_with_timings(self.comp, transcript)
        timings = {"stt_s": stt_s, **timings}

        if self.speak:
            try:
                body["audio_b64"], timings["tts_s"] = synthesize_b64(
                    self.comp, body["answer"]
                )
            except HTTPException:
                pass  # TTS unavailable; still surface the text answer

        timings["total_s"] = round(sum(timings.values()), 3)
        body["transcript"] = transcript
        body["timings"] = timings
        self.results.put(body)
# Schemas


class AskRequest(BaseModel):
    question: str
    speak: bool = False


class SpeakRequest(BaseModel):
    text: str


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

        app.state.wake_listener = None
        yield

        listener = app.state.wake_listener
        if listener is not None:
            listener.stop()

    app = FastAPI(title="PDVA service", lifespan=lifespan)

    def comp() -> Components:
        return app.state.comp

    @app.get("/health")
    def health():
        c = comp()
        ready = c.assistant.ready()

        return {
            "llm_ready": ready["llm"],
            "stt_ready": ready["stt"],
            "tts_ready": ready["tts"],
            "vision_ready": ready["vision"],
            "indexed_chunks": c.index.count(),
        }

    @app.get("/documents")
    def list_documents():
        files = sorted(
            p.name for p in DOCS_DIR.iterdir() if p.suffix.lower() in SUPPORTED_DOCS
        )

        return {"documents": files, "chunks": comp().index.count()}

    @app.post("/documents")
    def upload_documents(files: list[UploadFile] = File(...)):
        saved = []

        for f in files:
            name = Path(f.filename or "").name

            if not name or Path(name).suffix.lower() not in SUPPORTED_DOCS:
                raise HTTPException(
                    400,
                    f"Unsupported file type: {f.filename!r} "
                    f"(supported: {sorted(SUPPORTED_DOCS)})",
                )

            dest = DOCS_DIR / name

            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)

            saved.append(str(dest))

        t0 = time.perf_counter()
        chunks = comp().index.add_documents(saved)

        return {
            "added": [Path(p).name for p in saved],
            "chunks_added": chunks,
            "index_s": round(time.perf_counter() - t0, 3),
        }

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

        body, timings = answer_with_timings(comp(), req.question)

        if req.speak:
            body["audio_b64"], timings["tts_s"] = synthesize_b64(comp(), body["answer"])

        timings["total_s"] = round(sum(v for v in timings.values()), 3)
        body["timings"] = timings

        return body

    @app.post("/voice/ask")
    def voice_ask(audio: UploadFile = File(...), speak: bool = False):
        c = comp()
        transcriber = c.assistant.transcriber

        if transcriber is None:
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
            transcript = transcriber.transcribe(path)
            stt_s = round(time.perf_counter() - t0, 3)
        finally:
            os.remove(path)

        if not transcript.strip():
            return {
                "transcript": "",
                "answer": "",
                "sources": [],
                "timings": {"stt_s": stt_s, "total_s": stt_s},
            }

        body, timings = answer_with_timings(c, transcript)
        timings = {"stt_s": stt_s, **timings}

        if speak:
            body["audio_b64"], timings["tts_s"] = synthesize_b64(c, body["answer"])

        timings["total_s"] = round(sum(timings.values()), 3)
        body["transcript"] = transcript
        body["timings"] = timings

        return body

    @app.post("/vision/ask")
    def vision_ask(
        image: UploadFile = File(...), question: str = Form(""), speak: bool = False
    ):
        c = comp()
        if c.assistant.vision is None:
            raise HTTPException(503, "Vision not available (moondream not loaded)")

        suffix = Path(image.filename or "img.png").suffix.lower() or ".png"
        if suffix not in SUPPORTED_IMAGES:
            raise HTTPException(400, f"Unsupported image type: {suffix}")

        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            t0 = time.perf_counter()
            q = question.strip()
            if q:
                answer = c.assistant.answer_about_image(path, q)
            else:
                answer = c.assistant.vision.describe(path)
            vision_s = round(time.perf_counter() - t0, 3)
        finally:
            os.remove(path)

        body = {"answer": answer, "sources": []}
        timings = {"vision_s": vision_s}
        if speak:
            body["audio_b64"], timings["tts_s"] = synthesize_b64(c, answer)
        timings["total_s"] = round(sum(timings.values()), 3)
        body["timings"] = timings
        return body

    # -- speak -------------------------------------------------------------
    @app.post("/speak")
    def speak(req: SpeakRequest):
        if not req.text.strip():
            raise HTTPException(400, "Empty text")

        b64, _ = synthesize_b64(comp(), req.text)

        return Response(content=base64.b64decode(b64), media_type="audio/wav")

    # -- wake word -----------------------------------------------------------
    @app.post("/voice/wake/start")
    def wake_start(speak: bool = False):
        c = comp()

        if c.assistant.transcriber is None:
            raise HTTPException(503, "STT not available (whisper model not loaded)")

        listener = app.state.wake_listener

        if listener is None or not listener.running:
            listener = WakeListener(c, speak=speak)
            app.state.wake_listener = listener

        listener.start()

        return {"running": True, "state": listener.state}

    @app.post("/voice/wake/stop")
    def wake_stop():
        listener = app.state.wake_listener

        if listener is not None:
            listener.stop()

        return {"running": False}

    @app.get("/voice/wake/status")
    def wake_status():
        listener = app.state.wake_listener

        return {
            "running": listener.running if listener else False,
            "state": listener.state if listener else "idle",
            "score": listener.last_score if listener else None,
        }

    @app.get("/voice/wake/result")
    def wake_result():
        listener = app.state.wake_listener

        if listener is None:
            return {"ready": False}

        try:
            body = listener.results.get_nowait()
            body["ready"] = True
            return body
        except pyqueue.Empty:
            return {"ready": False}

    return app


app = create_app()
