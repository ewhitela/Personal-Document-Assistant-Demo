# End-to-end latency breakdown

Generated 2026-07-28 19:51 UTC by `bench_latency.py`.

## Configuration

- GPU: NVIDIA GeForce GTX 1080 Ti
- Platform: Linux-7.0.0-28-generic-x86_64-with-glibc2.39, Python 3.11.15
- LLM: `llama3.2:3b` via ollama (temperature 0.05)
- Embeddings: `all-MiniLM-L6-v2`, top-k 6, chunk size 200 words
- STT: faster-whisper `base.en` on cpu (int8)
- TTS: Piper `en_US-lessac-medium.onnx` (CUDA: False)
- 5 timed runs per question, 1 cold warm-up run(s) discarded
- Budget: 3 s from question to spoken answer

`stt` is measured on `test.wav`, repeated once per run. Transcription cost tracks utterance length rather than question text, so a fixed clip keeps it comparable across questions.

## Per-stage timings (all questions pooled)

| Stage | n | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stt | 50 | 509 | 512 | 483 | 564 | 545 |
| retrieve | 50 | 7 | 8 | 7 | 9 | 8 |
| generate | 50 | 359 | 545 | 164 | 1733 | 1376 |
| verify | 50 | 2 | 2 | 0 | 3 | 2 |
| tts | 50 | 215 | 248 | 62 | 763 | 735 |
| total | 50 | 1088 | 1315 | 794 | 3003 | 2088 |

## Per-question totals (median of runs)

| Question | stt (ms) | retrieve (ms) | generate (ms) | verify (ms) | tts (ms) | total (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| What rivers meet in Pittsburgh, and what do they form? | 495 | 7 | 360 | 2 | 219 | 1081 |
| What was Pittsburgh's historical industry, and how has th... | 502 | 8 | 566 | 2 | 492 | 1602 |
| What's Pittsburgh's population according to the most rece... | 521 | 8 | 172 | 2 | 124 | 826 |
| What's the elevation of Flagstaff, and why does that matt... | 497 | 7 | 735 | 2 | 743 | 1982 |
| Did Flagstaff launch a bid to host the 1960 Winter Olympics? | 489 | 7 | 380 | 2 | 184 | 1110 |
| How many boilers did Itsukushima have, and what was her a... | 494 | 7 | 328 | 2 | 259 | 1116 |
| Did Itsukushima survive World War II? | 497 | 7 | 328 | 2 | 232 | 1066 |
| What's the maximum recorded length of the black-faced ble... | 513 | 7 | 247 | 2 | 82 | 853 |
| Is the black-faced blenny venomous? | 539 | 8 | 324 | 2 | 120 | 990 |
| What is the capital of Andhra Pradesh? | 509 | 8 | 229 | 0 | 66 | 809 |

## Verdict

Median turn: **1.09 s**. p95: **2.09 s** — within the 3 s budget.

Cold first turn: **2.75 s** (includes ollama's one-time model load). Warm median is 1.09 s — the gap is load time, not steady-state latency.

`stt` dominates at 47% of the median turn.
