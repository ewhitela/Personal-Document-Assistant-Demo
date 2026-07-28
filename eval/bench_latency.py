"""Week 12: Latency benchmark for the finished assistant.

Produces the graded end-to-end latency breakdown. Unlike `evaluate.py`, which
times a single turn to stdout, this runs each question N times, discards a
warm-up (so ollama's model load does not land in `generate_s`), and writes a
markdown table plus the raw samples to disk.

The stages match what `service/app.py` reports, so the numbers here are directly
comparable to what the UI shows during the demo:

    stt -> retrieve -> generate -> verify -> tts

`stt` is measured on a fixed audio clip (default `test.wav`), because
transcription time tracks the length of the utterance, not the question text.

Usage:

    python bench_latency.py                      # 5 runs/question, TTS included

    python bench_latency.py --audio my_question.wav --out docs/latency.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pdva import DocumentIndex, LocalLLM, RAGPipeline, Speaker, Transcriber, config
from pdva.verifier import verify

BUDGET_S = 3.0
STAGES = ["stt", "retrieve", "generate", "verify", "tts"]

# Below this sample count, p95 interpolates too close to the observed max to
# be a stable tail estimate rather than a worst-case; always caveat it.
P95_CAVEAT_THRESHOLD = 50

DEFAULT_QUESTIONS = [
    "What rivers meet in Pittsburgh, and what do they form?",
    "What was Pittsburgh's historical industry, and how has that changed?",
    (
        "What's the elevation of Flagstaff, and why does that matter for "
        "the city's climate?"
    ),
    ("How many boilers did Itsukushima have, and what was her average maximum speed?"),
    "What's the maximum recorded length of the black-faced blenny?",
]


def load_questions(use_gold: bool) -> list[str]:
    """Take the question set from evaluate.py's GOLD if asked, else the default.

    Reusing GOLD keeps the latency numbers on the same questions as the quality
    numbers, so the two halves of the Week 12 report describe the same workload.
    """
    if not use_gold:
        return DEFAULT_QUESTIONS
    try:
        from evaluate import GOLD
    except ImportError as exc:  # pragma: no cover - depends on cwd
        print(
            f"warning: could not import GOLD from evaluate.py ({exc}); "
            "falling back to the built-in question set"
        )
        return DEFAULT_QUESTIONS
    return [q for q, _source, _must in GOLD]


# --- statistics ----------------------------------------------------------------


def percentile(samples: list[float], q: float) -> float:
    """Linear-interpolated percentile. q is a fraction, e.g. 0.95."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


@dataclass
class Samples:
    """Per-stage timing samples, in seconds."""

    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, stage: str, seconds: float) -> None:
        self.values.setdefault(stage, []).append(seconds)

    def summary(self, stage: str) -> dict | None:
        xs = self.values.get(stage)
        if not xs:
            return None
        return {
            "n": len(xs),
            "median": statistics.median(xs),
            "mean": statistics.fmean(xs),
            "min": min(xs),
            "max": max(xs),
            "p95": percentile(xs, 0.95),
        }


# --- component setup -----------------------------------------------------------


@dataclass
class Rig:
    """The modules under test, plus which stages are actually measurable."""

    rag: RAGPipeline
    transcriber: Transcriber | None
    speaker: Speaker | None
    audio_path: str | None
    notes: list[str] = field(default_factory=list)


def build_rig(audio: str | None, want_tts: bool) -> Rig:
    index = DocumentIndex()
    count = index.count()
    if count == 0:
        raise SystemExit(
            f"The index at {config.PERSIST_DIR} is empty. Index some documents "
            "first (via the Streamlit sidebar or DocumentIndex.add_documents)."
        )
    print(f"index: {count} chunks in {config.COLLECTION_NAME}")

    rag = RAGPipeline(index=index, llm=LocalLLM())
    if not rag.llm.is_ready():
        raise SystemExit(
            f"ollama is not serving {config.LLM_MODEL}. Start ollama and run "
            f"`ollama pull {config.LLM_MODEL}`."
        )

    notes: list[str] = []

    transcriber: Transcriber | None = None
    audio_path: str | None = None
    if audio is None:
        notes.append("STT skipped: --no-stt")
    elif not Path(audio).exists():
        notes.append(f"STT skipped: audio file not found ({audio})")
    else:
        try:
            transcriber = Transcriber()
        except Exception as exc:  # report the gap, do not abort the run
            notes.append(f"STT skipped: transcriber failed to load ({exc})")
        else:
            if transcriber.is_ready():
                audio_path = audio
            else:
                transcriber = None
                notes.append("STT skipped: faster-whisper model not loaded")

    speaker: Speaker | None = None
    if not want_tts:
        notes.append("TTS skipped: --no-tts")
    else:
        try:
            speaker = Speaker()
        except Exception as exc:
            notes.append(f"TTS skipped: speaker failed to load ({exc})")
        else:
            if not speaker.is_ready():
                speaker = None
                notes.append("TTS skipped: Piper voice not loaded")

    for note in notes:
        print(f"note: {note}")

    return Rig(
        rag=rag,
        transcriber=transcriber,
        speaker=speaker,
        audio_path=audio_path,
        notes=notes,
    )


# --- one timed turn ------------------------------------------------------------


def timed_turn(rig: Rig, question: str, tts_wav: str) -> dict:
    """Run one turn, timing each stage the way service/app.py does.

    NOTE: this calls index.search / build_prompt / llm.generate / verify
    directly rather than RAGPipeline.answer(), so it assumes answer() does
    nothing beyond wiring those four calls together. If that assumption is
    wrong, this benchmark under-reports whatever extra work answer() does.
    Confirm against rag.py before trusting these numbers as "what the demo
    actually runs."
    """
    timings: dict[str, float] = {}

    if rig.transcriber is not None and rig.audio_path is not None:
        t0 = time.perf_counter()
        rig.transcriber.transcribe(rig.audio_path)
        timings["stt"] = time.perf_counter() - t0

    rag = rig.rag

    t0 = time.perf_counter()
    passages = rag.index.search(question, rag.k)
    t1 = time.perf_counter()
    prompt = rag.build_prompt(question, passages)
    text = rag.llm.generate(prompt, system=rag.SYSTEM_PROMPT)
    t2 = time.perf_counter()
    raw_text = text
    text, _report = verify(text, passages, question)
    t3 = time.perf_counter()

    timings["retrieve"] = t1 - t0
    timings["generate"] = t2 - t1
    timings["verify"] = t3 - t2

    if rig.speaker is not None:
        t4 = time.perf_counter()
        rig.speaker.synthesize(text, tts_wav)
        timings["tts"] = time.perf_counter() - t4

    timings["total"] = sum(timings.values())
    return {
        "timings": timings,
        "answer": text,
        "raw_answer": raw_text,
        "n_sources": len(passages),
    }


def run_benchmark(rig: Rig, questions: list[str], runs: int, warmup: int) -> dict:
    overall = Samples()
    per_question: dict[str, Samples] = {}
    raw: list[dict] = []
    cold_start: dict[str, float] | None = None

    fd, tts_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        if warmup:
            print(f"warm-up: {warmup} discarded run(s)...")
            for i in range(warmup):
                turn = timed_turn(rig, questions[0], tts_wav)
                if i == 0:
                    # Only the very first turn is genuinely cold (ollama model
                    # load, first-call overhead etc). Keep it instead of
                    # throwing it away, since it's the number worth reporting
                    # separately from the warm median.
                    cold_start = turn["timings"]
                    raw.append(
                        {
                            "question": questions[0],
                            "run": "cold",
                            "timings": turn["timings"],
                        }
                    )

        for question in questions:
            samples = Samples()
            per_question[question] = samples
            print(f"\n{question}")
            for i in range(runs):
                turn = timed_turn(rig, question, tts_wav)
                t = turn["timings"]
                for stage, seconds in t.items():
                    samples.add(stage, seconds)
                    overall.add(stage, seconds)

                raw.append(
                    {
                        "question": question,
                        "run": i,
                        "timings": t,
                        "answer_chars": len(turn["answer"]),
                        "answer": turn["answer"],
                        "raw_answer": turn["raw_answer"],
                    }
                )

                flag = "  OVER BUDGET" if t["total"] > BUDGET_S else ""
                print(f"  run {i + 1}/{runs}  total {t['total'] * 1000:8.1f} ms{flag}")
    finally:
        Path(tts_wav).unlink(missing_ok=True)

    return {
        "overall": overall,
        "per_question": per_question,
        "raw": raw,
        "cold_start": cold_start,
    }


# --- reporting -----------------------------------------------------------------


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:  # best effort only
        return "unknown (nvidia-smi unavailable)"


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}"


def build_report(results: dict, rig: Rig, runs: int, warmup: int) -> str:
    overall: Samples = results["overall"]
    per_question: dict[str, Samples] = results["per_question"]
    cold_start: dict | None = results.get("cold_start")

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# End-to-end latency breakdown",
        "",
        f"Generated {stamp} by `bench_latency.py`.",
        "",
        "## Configuration",
        "",
        f"- GPU: {gpu_name()}",
        f"- Platform: {platform.platform()}, Python {platform.python_version()}",
        (
            f"- LLM: `{config.LLM_MODEL}` via ollama "
            f"(temperature {config.LLM_TEMPERATURE})"
        ),
        (
            f"- Embeddings: `{config.EMBEDDING_MODEL}`, "
            f"top-k {config.RAG_TOP_K}, chunk size {config.CHUNK_SIZE} words"
        ),
        (
            f"- STT: faster-whisper `{config.WHISPER_MODEL}` on "
            f"{config.WHISPER_DEVICE} ({config.WHISPER_COMPUTE})"
        ),
        f"- TTS: Piper `{config.PIPER_VOICE}` (CUDA: {config.PIPER_USE_CUDA})",
        f"- {runs} timed runs per question, {warmup} cold warm-up run(s) discarded",
        f"- Budget: {BUDGET_S:.0f} s from question to spoken answer",
        "",
    ]

    if rig.audio_path:
        lines += [
            (
                f"`stt` is measured on `{rig.audio_path}`, repeated once per "
                "run. Transcription cost tracks utterance length rather than "
                "question text, so a fixed clip keeps it comparable across "
                "questions."
            ),
            "",
        ]

    if rig.notes:
        lines += ["Stages not measured in this run:", ""]
        lines += [f"- {note}" for note in rig.notes]
        lines += [""]

    lines += [
        "## Per-stage timings (all questions pooled)",
        "",
        "| Stage | n | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | p95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for stage in [*STAGES, "total"]:
        s = overall.summary(stage)
        if s is None:
            continue
        lines.append(
            f"| {stage} | {s['n']} | {ms(s['median'])} | {ms(s['mean'])} | "
            f"{ms(s['min'])} | {ms(s['max'])} | {ms(s['p95'])} |"
        )

    total = overall.summary("total")
    lines += ["", "## Per-question totals (median of runs)", ""]
    header = "| Question |"
    divider = "| --- |"
    measured = [st for st in STAGES if overall.summary(st) is not None]
    for stage in [*measured, "total"]:
        header += f" {stage} (ms) |"
        divider += " ---: |"
    lines += [header, divider]

    for question, samples in per_question.items():
        short = question if len(question) <= 60 else question[:57] + "..."
        row = f"| {short} |"
        for stage in [*measured, "total"]:
            s = samples.summary(stage)
            row += f" {ms(s['median']) if s else '-'} |"
        lines.append(row)

    lines += ["", "## Verdict", ""]
    if total:
        within = "within" if total["p95"] <= BUDGET_S else "over"
        lines += [
            (
                f"Median turn: **{total['median']:.2f} s**. "
                f"p95: **{total['p95']:.2f} s** — {within} the "
                f"{BUDGET_S:.0f} s budget."
            ),
        ]
        if cold_start and "total" in cold_start:
            lines += [
                "",
                (
                    f"Cold first turn: **{cold_start['total']:.2f} s** "
                    f"(includes ollama's one-time model load). Warm median "
                    f"is {total['median']:.2f} s — the gap is load time, not "
                    "steady-state latency."
                ),
            ]
        n_total = len(overall.values.get("total", []))
        if n_total < P95_CAVEAT_THRESHOLD:
            lines += [
                "",
                (
                    f"With only {n_total} samples, p95 interpolates close to "
                    "the observed maximum; treat it as a worst-case rather "
                    "than a stable tail estimate."
                ),
            ]
        slowest = max(
            (st for st in measured),
            key=lambda st: overall.summary(st)["median"],
            default=None,
        )
        if slowest:
            share = overall.summary(slowest)["median"] / total["median"] * 100
            lines += [
                "",
                f"`{slowest}` dominates at {share:.0f}% of the median turn.",
            ]

    return "\n".join(lines) + "\n"


# --- entry point ---------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs", type=int, default=5, help="timed runs per question (default 5)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="discarded warm-up runs (default 1)"
    )
    parser.add_argument(
        "--audio",
        default="test.wav",
        help="wav clip used to time STT (default test.wav)",
    )
    parser.add_argument(
        "--no-stt", action="store_true", help="skip the transcription stage"
    )
    parser.add_argument(
        "--no-tts", action="store_true", help="skip the speech synthesis stage"
    )
    parser.add_argument(
        "--gold", action="store_true", help="use the GOLD question set from evaluate.py"
    )
    parser.add_argument(
        "--out",
        default="latency_results.md",
        help="markdown report path (default latency_results.md)",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="also write raw per-run samples as JSON",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    questions = load_questions(args.gold)
    rig = build_rig(None if args.no_stt else args.audio, want_tts=not args.no_tts)

    results = run_benchmark(rig, questions, args.runs, max(0, args.warmup))
    report = build_report(results, rig, args.runs, max(0, args.warmup))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nwrote {out}")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results["raw"], indent=2), encoding="utf-8")
        print(f"wrote {json_path}")

    total = results["overall"].summary("total")
    return 0 if total and total["p95"] <= BUDGET_S else 1


if __name__ == "__main__":
    sys.exit(main())
