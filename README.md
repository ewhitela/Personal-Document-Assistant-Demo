# Personal Document Voice Assistant (PDVA)

Speak a question, get a spoken answer grounded in your own documents. Runs
entirely locally: faster-whisper (STT) → ChromaDB + all-MiniLM-L6-v2
(retrieval) → llama3.2:3b via ollama (generation) → Piper (TTS), orchestrated
by a FastAPI service with a Streamlit UI.

## Layout

pdva/
types.py shared dataclasses: Passage, RAGAnswer, TranscriptSegment
config.py shared settings: paths, model names, chunk sizes
embedding_index.py Week 4: DocumentIndex (embedding + retrieval)
llm.py Week 5: LocalLLM (ollama wrapper)
rag.py Week 6: RAGPipeline (grounded answering)
transcriber.py Week 7: Transcriber (speech-to-text)
tts.py Week 8: Speaker (text-to-speech)
vision.py Week 9: VisionModel (optional, image questions)

service/
app.py Week 10: FastAPI service orchestrating all modules

ui/
streamlit_app.py Week 10: Streamlit interface (thin client of the service)

tests/
test_week4_index.py ... test_week10_service.py


## Setup

Requires Python 3.11.

python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Install the ollama app from https://ollama.com, then pull the model:

ollama pull llama3.2:3b

Download the Piper voice into the project root (the .onnx and .onnx.json
files must sit next to each other):

python -m piper.download_voices en_US-lessac-medium

STT and TTS run on CPU by design; the GPU is dedicated to the LLM. No CUDA
setup is required beyond ollama's own.

## Running the assistant

Two terminals, both with the venv activated, from the project root:

uvicorn service.app:app --port 8080 # terminal 1: the service
streamlit run ui/streamlit_app.py # terminal 2: the UI


Open http://localhost:8501. In the sidebar, upload .txt/.md/.pdf documents
and click "Index uploads". Then ask a question — by typing, or by recording
in the Voice tab (requires a browser with microphone access; Streamlit
>= 1.39). Toggle "Speak answers" to hear responses read aloud.

Each answer shows its sources and a per-stage latency breakdown
(stt / retrieve / generate / tts) against the 3-second budget.

## Tests

python tests/test_week4_index.py # ... through test_week9_vision.py
python -m pytest tests/test_week10_service.py -q


PASS/TODO/SKIP/WARN/FAIL semantics: SKIP means a prerequisite is missing
(e.g. ollama not running) — the suites are safe to run anytime. The Week 10
tests use fake components and need no models.

## Troubleshooting

- Sidebar says "Service unreachable": terminal 1 isn't running, or crashed —
  check its output.
- Transcript comes back empty or as "You": the recording is silent — check
  browser mic permissions and OS input device/gain.
- `LLM ❌` in the sidebar: ollama isn't running or the model isn't pulled.