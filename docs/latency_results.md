# End-to-end latency breakdown

Generated 2026-07-27 18:27 UTC by `bench_latency.py`.

## Configuration

- GPU: NVIDIA GeForce GTX 1080 Ti
- Platform: Linux-7.0.0-28-generic-x86_64-with-glibc2.39, Python 3.11.15
- LLM: `llama3.2:3b` via ollama (temperature 0.05)
- Embeddings: `all-MiniLM-L6-v2`, top-k 7, chunk size 200 words
- STT: faster-whisper `base.en` on cpu (int8)
- TTS: Piper `en_US-lessac-medium.onnx` (CUDA: False)
- 5 timed runs per question, 1 warm-up run(s) discarded
- Budget: 3 s from question to spoken answer

`stt` is measured on `test.wav`, repeated once per run. Transcription cost tracks utterance length rather than question text, so a fixed clip keeps it comparable across questions.

## Per-stage timings (all questions pooled)

| Stage | n | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stt | 25 | 471 | 473 | 451 | 505 | 502 |
| retrieve | 25 | 8 | 8 | 7 | 9 | 8 |
| generate | 25 | 381 | 461 | 164 | 790 | 751 |
| verify | 25 | 2 | 3 | 2 | 3 | 3 |
| tts | 25 | 239 | 306 | 55 | 597 | 577 |
| total | 25 | 1072 | 1250 | 693 | 1865 | 1822 |

## Per-question totals (median of runs)

| Question | stt (ms) | retrieve (ms) | generate (ms) | verify (ms) | tts (ms) | total (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| What rivers meet in Pittsburgh, and what do they form? | 500 | 8 | 381 | 3 | 173 | 1048 |
| What was Pittsburgh's historical industry, and how has th... | 467 | 7 | 708 | 2 | 514 | 1696 |
| What's the elevation of Flagstaff, and why does that matt... | 492 | 8 | 751 | 2 | 570 | 1818 |
| How many boilers did Itsukushima have, and what was her a... | 452 | 7 | 339 | 3 | 184 | 984 |
| What's the maximum recorded length of the black-faced ble... | 471 | 7 | 173 | 2 | 57 | 703 |

## Verdict

Median turn: **1.07 s**. p95: **1.82 s** — within the 3 s budget.

`stt` dominates at 44% of the median turn.
