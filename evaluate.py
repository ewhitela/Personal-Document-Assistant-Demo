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
        REFUSAL_PHRASE,
    ),
    (
        "What's the maximum recorded length of the black-faced blenny?",
        "Black-faced blenny - Wikipedia.pdf",
        "8 cent",
    ),
    (
        "Is the black-faced blenny venomous?",
        "Black-faced blenny - Wikipedia.pdf",
        REFUSAL_PHRASE,
    ),
    ("What is the capital of Andhra Pradesh?", None, REFUSAL_PHRASE),
]


def evaluate_qa(assistant) -> dict:
    """Run the GOLD set through the assistant and score it.

    Behavior:
        For each (question, expected_source, must_contain) in GOLD:
            result = assistant.answer_text(question)
            - retrieval hit: expected_source is None, or expected_source is among
              [s.source for s in result.sources].
            - answer ok: must_contain (lowercased) appears in result.answer
              lowercased. For a None expected_source, "ok" means the assistant
              refused (your refusal phrase appears).
        Return a dict of counts, for example
            {"n": ..., "retrieval_hits": ..., "answer_ok": ...}
        and print a short summary.
    """
    n = len(GOLD)
    retrieval_hits = 0
    answer_ok = 0
    rows = []

    for question, expected_source, must_contain in GOLD:
        result = assistant.answer_text(question)
        answer_lower = result.answer.lower()
        sources = [s.source for s in result.sources]

        if expected_source is None:
            retrieval_hit = True  # no source expected; nothing to check
        else:
            retrieval_hit = expected_source in sources

        hit_ok = must_contain.lower() in answer_lower

        retrieval_hits += int(retrieval_hit)
        answer_ok += int(hit_ok)

        rows.append((question, retrieval_hit, hit_ok))

    print(f"\n{'question':<70} {'retrieval':<10} {'answer'}")
    for question, retrieval_hit, hit_ok in rows:
        q_display = (question[:67] + "...") if len(question) > 70 else question
        print(
            f"{q_display:<70} {'OK' if retrieval_hit else 'MISS':<10} "
            f"{'OK' if hit_ok else 'MISS'}"
        )

    summary = {"n": n, "retrieval_hits": retrieval_hits, "answer_ok": answer_ok}
    print(f"\nRetrieval: {retrieval_hits}/{n}   Answer quality: {answer_ok}/{n}")
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
