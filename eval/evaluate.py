"""Week 12: A small evaluation harness for the finished assistant.

Two things you want numbers for on demo day:
    1. Latency: how long each stage of a turn takes, and the total.
    2. Answer quality: on a small set of questions with known answers, does the
       assistant retrieve the right source and answer (or correctly refuse)?

measure_latency is provided and works as-is. You fill in the question set and the
grading in evaluate_qa. Run it against your real, built assistant:

    python eval/evaluate.py
"""

from __future__ import annotations

import time
from collections.abc import Callable


def measure_latency(label: str, fn: Callable, *args, **kwargs):
    """Time a single call. Returns (result, seconds). Provided, works as-is."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{label:<14} {elapsed * 1000:8.1f} ms")
    return result, elapsed


def time_text_turn(assistant, question: str):
    """Time one typed turn end to end. Provided as an example of measure_latency."""
    print(f"\nQuestion: {question!r}")
    result, total = measure_latency("answer_text", assistant.answer_text, question)
    print(f"{'total':<14} {total * 1000:8.1f} ms")
    return result


def time_text_turn_by_stage(assistant, question: str, speak: bool = False):
    """Time retrieve, generate, and (optionally) synthesize separately.

    answer_text() times the whole turn as one call, which hides where the
    time actually goes. This calls the pipeline's own stages directly the
    same way service/app.py's answer_with_timings does, so retrieve_s and
    generate_s are visible on their own, not just folded into one total.
    """
    print(f"\nQuestion: {question!r}")
    pipeline = assistant.rag
    t0 = time.perf_counter()
    passages = pipeline.index.search(question, pipeline.k)
    t1 = time.perf_counter()
    prompt = pipeline.build_prompt(question, passages)
    text = pipeline.llm.generate(prompt, system=pipeline.SYSTEM_PROMPT)
    t2 = time.perf_counter()

    retrieve_s = t1 - t0
    generate_s = t2 - t1
    print(f"{'retrieve':<14} {retrieve_s * 1000:8.1f} ms")
    print(f"{'generate':<14} {generate_s * 1000:8.1f} ms")

    tts_s = None
    if speak:
        if assistant.speaker is None:
            print("  (speak requested but no speaker configured; skipping tts timing)")
        else:
            t3 = time.perf_counter()
            assistant.speaker.synthesize(text, "/tmp/_eval_tts.wav")
            tts_s = time.perf_counter() - t3
            print(f"{'synthesize':<14} {tts_s * 1000:8.1f} ms")

    total_s = retrieve_s + generate_s + (tts_s or 0.0)
    print(
        f"{'total':<14} {total_s * 1000:8.1f} ms"
        f"{'  *** OVER 3s BUDGET ***' if total_s > 3.0 else ''}"
    )

    return {
        "answer": text,
        "sources": passages,
        "retrieve_s": retrieve_s,
        "generate_s": generate_s,
        "tts_s": tts_s,
        "total_s": total_s,
    }


# --- You fill these in for the quality evaluation -------------------------------

# A tiny gold set: each item is (question, expected_source, must_contain).
# Add 5 to 10 questions whose answers you know from your own documents, plus at
# least one whose answer is NOT in your documents (expected_source = None) to
# check that the assistant refuses instead of inventing.
#
# Drawn from this week's diagnostic session (see rag_eval_report_2026-07-23.md
# for the full evaluation history and failure taxonomy this set is built on).
REFUSAL_PHRASE = "i don't know based on your documents"

GOLD: list[tuple] = [
    (
        "What rivers meet in Pittsburgh, and what do they form?",
        "Pittsburgh - Wikipedia.pdf",
        "ohio river",
    ),
    (
        "What was Pittsburgh's historical industry, and how has that changed?",
        "Pittsburgh - Wikipedia.pdf",
        "steel",
    ),
    (
        (
            "What's Pittsburgh's population according to the most recent census "
            "mentioned in the article?"
        ),
        "Pittsburgh - Wikipedia.pdf",
        "302,971",
    ),
    (
        (
            "What's the elevation of Flagstaff, and why does that matter for the "
            "city's climate?"
        ),
        "Flagstaff, Arizona - Wikipedia.pdf",
        "7,000",
    ),
    (
        "Has Flagstaff ever hosted a Winter Olympics?",
        "Flagstaff, Arizona - Wikipedia.pdf",
        "1960",
    ),
    (
        (
            "How many boilers did Itsukushima have, and what was her average "
            "maximum speed?"
        ),
        "Japanese cruiser Itsukushima - Wikipedia.pdf",
        "16.78",
    ),
    (
        "Did Itsukushima survive World War II?",
        "Japanese cruiser Itsukushima - Wikipedia.pdf",
        "1926",
    ),
    (
        "What's the maximum recorded length of the black-faced blenny?",
        "Black-faced blenny - Wikipedia.pdf",
        "8 cent",
    ),
    (
        "Is the black-faced blenny venomous?",
        "Black-faced blenny - Wikipedia.pdf",
        (REFUSAL_PHRASE, "does not state", "not mentioned", "not venomous"),
    ),
    ("What is the capital of Andhra Pradesh?", None, REFUSAL_PHRASE),
]


def evaluate_qa(assistant, k: int = 5) -> dict:
    """Run the GOLD set through the assistant k times each and score it.

    Answers are sampled, not deterministic, so a single run per question cannot
    distinguish a solid pass from a borderline one that happens to land well.
    Each question is asked k times and scored as a pass rate.

    Behavior:
        For each (question, expected_source, must_contain) in GOLD, ask the
        question k times:
        - retrieval hit: expected_source is None, or expected_source is among
          [s.source for s in result.sources].
        - answer ok: must_contain (lowercased) appears in result.answer
          lowercased. For a None expected_source, "ok" means the assistant
          refused (the refusal phrase appears).

    A question counts toward retrieval_hits / answer_ok only if it passed on
    every one of the k runs. Anything in between is reported as flaky.
    """
    n = len(GOLD)
    retrieval_hits = 0
    answer_ok = 0
    rows = []
    flaky = []

    for question, expected_source, must_contain in GOLD:
        r_hits = 0
        a_hits = 0

        for _ in range(k):
            result = assistant.answer_text(question)

            if expected_source is None:
                r_hits += 1  # no source expected; nothing to check
            elif expected_source in [s.source for s in result.sources]:
                r_hits += 1

            if isinstance(must_contain, str):
                accepted = (must_contain,)
            else:
                accepted = must_contain

            if any(phrase.lower() in result.answer.lower() for phrase in accepted):
                a_hits += 1

        retrieval_hits += int(r_hits == k)
        answer_ok += int(a_hits == k)
        rows.append((question, r_hits, a_hits))

        if 0 < a_hits < k:
            flaky.append((question, a_hits))

    print(f"\n{'question':<70} {'retrieval':<11} {'answer'}")
    for question, r_hits, a_hits in rows:
        q_display = (question[:67] + "...") if len(question) > 70 else question
        print(f"{q_display:<70} {f'{r_hits}/{k}':<11} {a_hits}/{k}")

    summary = {
        "n": n,
        "k": k,
        "retrieval_hits": retrieval_hits,
        "answer_ok": answer_ok,
        "answer_rate": sum(a for _q, _r, a in rows) / (n * k),
        "flaky": [q for q, _a in flaky],
    }

    print(
        f"\nPassed all {k} runs -- retrieval: {retrieval_hits}/{n}   "
        f"answer quality: {answer_ok}/{n}"
    )
    print(f"Mean answer pass rate: {summary['answer_rate']:.0%}")

    if flaky:
        print(f"\nFlaky ({len(flaky)}): passed some runs but not all")
        for question, a_hits in flaky:
            q_display = (question[:67] + "...") if len(question) > 70 else question
            print(f"  {a_hits}/{k}  {q_display}")

    return summary


if __name__ == "__main__":
    # Wire your real assistant here, then uncomment.
    #
    from pdva import (
        Assistant,
        DocumentIndex,
        LocalLLM,
        RAGPipeline,
        Speaker,
        Transcriber,
    )

    index = DocumentIndex()  # loads existing PERSIST_DIR / COLLECTION_NAME
    bot = Assistant(Transcriber(), RAGPipeline(index=index, llm=LocalLLM()), Speaker())
    time_text_turn(bot, "What rivers meet in Pittsburgh, and what do they form?")
    time_text_turn_by_stage(
        bot, "What rivers meet in Pittsburgh, and what do they form?"
    )
    print(evaluate_qa(bot))
