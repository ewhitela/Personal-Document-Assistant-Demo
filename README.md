# Personal Document Voice Assistant (PDVA)

Speak a question, get a spoken answer grounded in your own documents. Runs
entirely locally: faster-whisper (STT) → ChromaDB + all-MiniLM-L6-v2
(retrieval) → llama3.2:3b via ollama (generation) → Piper (TTS), orchestrated
by a FastAPI service with a Streamlit UI.

## Hardware assumptions

Tested on an NVIDIA GTX 1080 Ti (11GB VRAM, Pascal architecture). The GPU is
used only by ollama for LLM inference; STT and TTS run on CPU by design, so
no CUDA setup is needed in this project's own venv. If you use a different
NVIDIA GPU generation, note that Pascal requires cuDNN 8 — cuDNN 9 dropped
support for compute capability < 7.5.

## Layout

    pdva/
      types.py              shared dataclasses: Passage, RAGAnswer, TranscriptSegment
      config.py              shared settings: paths, model names, chunk sizes
      embedding_index.py     Week 4: DocumentIndex (embedding + retrieval)
      llm.py                 Week 5: LocalLLM (ollama wrapper)
      rag.py                 Week 6: RAGPipeline (grounded answering)
      transcriber.py         Week 7: Transcriber (speech-to-text)
      tts.py                 Week 8: Speaker (text-to-speech)
      vision.py              Week 9: VisionModel (optional, image questions)
      assistant.py           Week 10: Assistant (owns the loaded models, one per process)
      verifier.py            Week 10: VerifiedRAGPipeline (flags ungrounded claims)

    service/
      app.py                 Week 10: FastAPI service orchestrating all modules

    ui/
      streamlit_app.py       Week 10: Streamlit interface (thin client of the service)

    demo_voice.py            Standalone capture + wake word + STT sanity check
                             (mic → openWakeWord "jarona" → VAD → faster-whisper).
                             Prints the recognized question only — does not call
                             RAGPipeline/LocalLLM/Speaker. Useful for verifying the
                             mic, wake word, and STT independently of the full service.

    eval/
      evaluate.py            Retrieval and answer-quality scoring
      bench_latency.py       Per-stage latency benchmark (the Week 12 deliverable)

    tests/
      test_week4_index.py ... test_week10_assistant.py

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Dependencies are
locked in `uv.lock`, which resolves both Linux and macOS from one file — the
same command works on the workstation and the laptop:

    uv sync --frozen              # core system
    uv sync --frozen --extra voice   # ...plus the wake-word loop

`--frozen` installs exactly what is locked and fails if `uv.lock` has drifted
from `pyproject.toml`. Regenerate the lock on the **Linux workstation only**
(`uv lock`); uv resolves every platform from that one run, so committing a
Mac-generated lock will silently drop the CUDA-side pins.

On Linux, torch comes from the cu118 index because the 1080 Ti is Pascal
(SM 6.1) and needs the cuDNN 8 build. On macOS that pin does not apply and uv
takes CPU/MPS wheels from PyPI.

Download the wake-word feature-extraction models (not bundled with the
openWakeWord pip package — required before `demo_voice.py` or any wake-word
code will run):

    python -c "import openwakeword.utils; openwakeword.utils.download_models()"

Install the ollama app from https://ollama.com, then pull the model:

    ollama pull llama3.2:3b

Download the Piper voice into the project root (the .onnx and .onnx.json
files must sit next to each other):

    python -m piper.download_voices en_US-lessac-medium

STT and TTS run on CPU by design; the GPU is dedicated to the LLM. No CUDA
setup is required beyond ollama's own.

## Running the assistant

Two terminals, from the project root:

    uv run uvicorn service.app:app --port 8080     # terminal 1: the service
    uv run streamlit run ui/streamlit_app.py       # terminal 2: the UI

Open http://localhost:8501. In the sidebar, upload .txt/.md/.pdf documents
and click "Index uploads". Then ask a question — by typing, or by recording
in the Voice tab (requires a browser with microphone access; Streamlit
>= 1.39). Toggle "Speak answers" to hear responses read aloud.

Each answer shows its sources and a per-stage latency breakdown
(stt / retrieve / generate / tts) against the 3-second budget.

To sanity-check the mic, wake word, and STT independently of the full
service:

    python demo_voice.py

## Running across two machines

For a demo where the mic is on a laptop and the models stay on the
workstation. The UI is a thin HTTP client, so only the service needs the GPU.

Workstation — bind to the LAN instead of loopback:

    uv run uvicorn service.app:app --host 0.0.0.0 --port 8080
    sudo ufw allow 8080/tcp        # if ufw is active

Laptop — point the UI at the workstation:

    PDVA_API_URL=http://<workstation-ip>:8080 uv run streamlit run ui/streamlit_app.py

ollama stays on `127.0.0.1`; only the FastAPI service is exposed. No CORS
configuration is needed, because Streamlit calls the API server-side.

Caveat for the latency deliverable: `total_s` is measured inside the service
and therefore excludes the audio upload and the TTS download. Over the lab
network that round-trip is real. Record wall-clock time in the UI as well and
report the difference as a separate `network_s` stage rather than folding it
into `total_s`.

## Tests

    python tests/test_week4_index.py   # ... through test_week9_vision.py
    uv run pytest tests/test_week10_assistant.py -q

PASS/TODO/SKIP/WARN/FAIL semantics: SKIP means a prerequisite is missing
(e.g. ollama not running) — the suites are safe to run anytime. The Week 10
tests use fake components and need no models.

## Troubleshooting

- Sidebar says "Service unreachable": terminal 1 isn't running, or crashed —
  check its output.
- Transcript comes back empty or as "You": the recording is silent — check
  browser mic permissions and OS input device/gain.
- `LLM ❌` in the sidebar: ollama isn't running or the model isn't pulled.
- `NoSuchFile` error loading `melspectrogram.onnx` (openWakeWord): the
  feature-extraction models weren't downloaded — run the
  `download_models()` command in Setup above.