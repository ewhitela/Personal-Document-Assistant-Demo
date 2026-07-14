

# PDVA Starter Kit (Weeks 4 to 9)

The code scaffold for the Personal Document Voice Assistant capstone. You implement
three modules across Weeks 4 to 6. The function and method signatures are fixed, so
your modules plug into each other (and into the Week 10 service) without changes.

## Layout

```
pdva/
  types.py            shared dataclasses: Passage, RAGAnswer
  config.py           shared settings: paths, model names, chunk sizes
  embedding_index.py  Week 4: DocumentIndex   (implement this first)
  llm.py              Week 5: LocalLLM
  rag.py              Week 6: RAGPipeline
  transcriber.py      Week 7: Transcriber     (speech-to-text)
  tts.py              Week 8: Speaker         (text-to-speech)
  vision.py           Week 9: VisionModel     (optional, image questions)
tests/
  test_week4_index.py
  test_week5_llm.py
  test_week6_rag.py
```

## Setup

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For Week 5, also install the ollama app from https://ollama.com, then pull a model:

```
ollama pull llama3.1:8b
```

## How to work

1. Open the module for the week (start with `pdva/embedding_index.py`).
2. Replace each `raise NotImplementedError` with your implementation. Read the
   docstrings first: they define the inputs, outputs, and required behavior. Do
   not change the signatures.
3. Run the matching test until everything is PASS:

```
python tests/test_week4_index.py
python tests/test_week5_llm.py
python tests/test_week6_rag.py
python tests/test_week7_stt.py
python tests/test_week8_tts.py
python tests/test_week9_vision.py
```

## Reading the test output

```
PASS   the check passed
TODO   you have not implemented this part yet
SKIP   a prerequisite is missing (for example, ollama is not running)
WARN   a soft behavior check; read the printed note, it may still be fine
FAIL   something is wrong; fix it
```

Full week-by-week instructions are in the Week 4, Week 5, and Week 6 handouts.

