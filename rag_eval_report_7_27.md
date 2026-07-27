# RAG evaluation report

Running log of retrieval and answer-quality evaluation for the Personal Document
Voice Assistant. Each dated entry records what was measured, what failed, and
what changed as a result. `eval/evaluate.py` cites this document as the source
of its GOLD set and failure taxonomy.

**Corpus under test:** 323 chunks (200 words, 50-word overlap) from five
Wikipedia PDFs — Pittsburgh, Flagstaff AZ, Japanese cruiser Itsukushima,
black-faced blenny, and one distractor.
**Stack:** `all-MiniLM-L6-v2` (384-dim, cosine) → ChromaDB, top-k 7 →
`llama3.2:3b` via ollama (temperature 0.05, `num_predict` 220) → `verifier.py`.

---

## Failure taxonomy

Identifiers used throughout this log.

| ID | Class | Description |
| --- | --- | --- |
| F1 | Over-strict verification | A correct, grounded answer is stripped or replaced with the abstain string by `verifier.py`. |
| F2 | Over-broad grounding rule | `SYSTEM_PROMPT` forbids an inference the passages plainly support, producing a refusal instead of an answer. |
| F3 | Refusal formatting | The model abstains correctly but paraphrases instead of emitting the exact abstain sentence. |
| F4 | Retrieval dilution | A semantically overloaded query term crowds the answering passage out of top-k. |
| F5 | Index contamination | Reference lists, citation fragments, or bare URLs occupy retrieval slots. |
| F6 | Answer drift | The model grounds itself in a real but irrelevant retrieved fact. |
| F7 | Measurement artifact | The harness measures something other than what the user experiences. |

---

## 2026-07-23 — Initial evaluation and GOLD set construction

> **TODO:** fill in from the 07-23 session. What this entry should cover:
> how the ten GOLD questions were chosen; why two refusal cases and one
> out-of-corpus case were included; the failure modes that motivated the
> citation-leakage, conflation, and comparison-inversion checks in
> `verifier.py` v2; and the scores at that point.

The GOLD set established here is ten questions across four documents:
five extractive, three open-ended, and two whose answers are absent from the
corpus (expected behaviour: refusal). Fields are
`(question, expected_source, must_contain)`, with `expected_source = None`
for the out-of-corpus case.

---

## 2026-07-27 — Verification regression, prompt tuning, and latency

### Summary

| Metric | Value |
| --- | --- |
| Retrieval (passed all 5 runs) | 10/10 |
| Answer quality (passed all 5 runs) | 8/10 |
| Mean answer pass rate | 80% |
| Flaky questions | 0 |

Both remaining misses are F3: the assistant abstains correctly but does not
emit the exact string. No question produced a wrong or ungrounded answer.

### Determinism

`evaluate_qa` was changed to sample each question `k=5` times and report a pass
rate rather than a boolean, because single-sample scoring cannot distinguish a
solid pass from a borderline case that happened to land well. Every question
scored either 5/5 or 0/5 with no intermediate results, so at temperature 0.05
decoding is effectively deterministic on this set and a score change can be
attributed to the change that caused it. Retrieval is exactly deterministic, as
expected for fixed embeddings.

### F1 — Verifier collapses correct answers to the abstain string

**Symptom.** "What was Pittsburgh's historical industry, and how has that
changed?" returned `"I don't know based on your documents."` on 5 of 5 runs
through the service path, while `evaluate.py` scored it 5/5 correct.

**Detection.** Instrumenting `bench_latency.py` to record both the raw
generation and the post-verification text. `raw_answer` was a full, correct
answer every run; `answer` was the refusal every run. Generation time of
~700 ms for a 37-character output was the first signal that something was
discarding work: the model was writing ~250 characters and the pipeline was
throwing them away.

**Mechanism.**

1. `SYSTEM_PROMPT` instructs the model to open with the specific figure or
   name asked for, so the answer begins with a one-word sentence: `"Steel."`
2. The second sentence contains `Pittsburgh's` in a non-initial position.
   `_entity_anchors` skips `words[0]`, so the same token in sentence-initial
   position is ignored — the bug only fires on mid-sentence occurrences.
3. `_anchors_present` built its lookup key as `a.rstrip(".-'").lower()`,
   yielding `pittsburgh's`. Passage word sets contained `pittsburgh`. No
   passage could satisfy the anchor.
4. The numbers `20` and `1980` were therefore classed as **conflated** (real,
   but not co-located with their entity), and the sentence was stripped.
5. The remainder, `"Steel."`, is 6 characters — below the 20-character floor
   in `verify()` — so the entire answer was replaced with `ABSTAIN`.

An asymmetry kept this invisible to the entity check: `grounding_words`
includes the question text, which contains `Pittsburgh's`, so
`_ungrounded_entities` reported nothing. `passage_words` does not include the
question. An anchor can be globally grounded and never locally co-locatable.

**Fix.** A single normalizer applied at all four key-construction sites
(`passage_words`, `grounding_words`, `_anchors_present`,
`_ungrounded_entities`):

```python
def _key(word: str) -> str:
    k = word.rstrip(".-'").lower()
    return k[:-2] if k.endswith("'s") else k
```

Partial application does not work: normalizing the sets without normalizing
the lookup leaves the same mismatch in place.

**Why it matters.** This is the most severe class of failure in the system.
It does not produce a wrong answer — it produces *no* answer, while every
observable signal reports success. The service returns HTTP 200, the UI shows
a valid latency breakdown, and `verify()` reports `abstained=True`, which
reads as the safety mechanism working as designed.

**Interaction, not a single defect.** Neither change was wrong alone. The
prompt rule improved three of five benchmark questions by putting the
requested figure first. The verifier's sub-20-character floor is a reasonable
guard against surfacing a fragment. Together they mean any single stripped
sentence following a short lead sentence collapses the whole answer to a
refusal. The floor should be revisited now that short opening sentences are
the intended style.

### F7 — The eval scored text the user never hears

`evaluate.py` calls `assistant.answer_text()`, which returns
`RAGPipeline.answer()` output directly. `service/app.py` returns the output of
`verify()`. F1 survived undetected because the harness measured the pipeline
before the stage that was breaking it.

**Action:** wrap the pipeline in `VerifiedRAGPipeline` before constructing the
`Assistant` in `evaluate.py`, so quality is scored on the served artifact.

### F2 — Grounding rule blocked a supported inference

**Symptom.** "Has Flagstaff ever hosted a Winter Olympics?" refused, 0/5.

This initially looked like a retrieval failure. It was not: the top-ranked
chunk contains "the city tried to launch a bid to be the host city of the 1960
Winter Olympics". The model had the sentence and refused anyway.

**Cause.** `SYSTEM_PROMPT` rule 1 read "do not add any claim the passages do
not state ... even if this seems like a reasonable assumption or is common
knowledge." Answering "no" requires inferring that a failed bid implies no
hosting. The passage never states Flagstaff did not host. The model followed
the instruction correctly; the instruction was too broad.

**Fix.** Rule 1 narrowed to currency and status claims only — its actual
purpose — with an explicit carve-out that it does not block stating a given
fact or drawing the plain conclusion that fact supports. A yes/no clause was
added to the length block so bare "no" answers carry their supporting detail.
The question moved to 5/5.

**Diagnostic note.** The retrieval column read OK throughout, because the
GOLD check tests whether `expected_source` appears in
`[s.source for s in result.sources]` — document level, not passage level.
Every Flagstaff question retrieves Flagstaff chunks. The check cannot
distinguish "found the right document" from "found the right passage" and
should be treated as a weak signal.

### F3 — Correct abstentions, non-exact wording

Two questions abstain correctly but paraphrase:

| Question | Output |
| --- | --- |
| Did Itsukushima survive World War II? | "No information about Itsukushima's fate during World War II is provided in the context passages." |
| Is the black-faced blenny venomous? | (exact string when sampled directly; scored 0/5 in the harness) |

Both are correct on the merits — neither article addresses the question. The
defect is cosmetic but not harmless: `verify()` gates its abstain fast-path on
`answer.strip() == ABSTAIN`, so a paraphrased refusal skips it and is run
through the number and entity checks instead, and `rep.abstained` never fires.

Scored strictly for now: the exact string is load-bearing, so a paraphrase is
a real if minor defect. Recorded as 8/10 rather than normalising the check.

### F4 — Retrieval dilution

For "Has Flagstaff ever hosted a Winter Olympics?", the retrieved set was
dominated by climate content — snowfall totals, temperature averages — because
*winter* is the semantically dominant term. The single Olympics-related chunk
that surfaced independently was about a 2016 athlete training facility, 56
years from the relevant event. Scores were flat (0.64 top, 0.50–0.55 for the
rest), indicating no strong match anywhere in the index.

The answering sentence was present, but embedded mid-chunk in climate prose.
Smaller chunks would isolate it; the tradeoff against context sufficiency is
untested.

### F5 — Index contamination

Retrieval for the Pittsburgh industry question returned, in the top seven:

- a newspaper citation line (*Pittsburgh Press*, April 14, 1982)
- a bare Google Books URL fragment

Two of seven slots on non-content chunks. The Wikipedia reference-stripping
and `is_content_chunk` filter are not catching citation-dense material that
appears inline rather than under a References heading.

### F6 — Answer drift

One run of the Pittsburgh industry question produced: "The steel industry
represented in the 1970s accounted for a higher share of the region's total
employment than the medical sector does today." Grounded, verified clean, and
not an answer to the question. Occasional behaviour of the 3B model latching
onto a real but tangential retrieved fact. Not addressed; recorded as a known
limitation.

### F7 — Prompt caching inflated the latency figures

`bench_latency.py` originally discarded one global warm-up. Because it rebuilds
an identical prompt for every run of a question, runs 2..n hit ollama's prompt
cache and paid almost no prefill — roughly 1.3 s cheaper than the first ask.
Only the first question benefited from the global warm-up, so every other
question's run 0 sat ~1.3 s above its own median.

Fixed by discarding a warm-up *per question* and recording cold runs
separately. Both figures are now reported: warm describes steady-state
pipeline cost, cold describes what a user experiences asking a question for
the first time.

---

## Latency

> **TODO:** regenerate after the verifier fix. The figures below were measured
> while the Pittsburgh industry question was still abstaining, so its
> `tts` and `total` are understated. Re-run:
> `python bench_latency.py --runs 20 --json latency_raw.json`

Budget: 3 s from question to spoken answer.

| Stage | Median (ms) | p95 (ms) | Notes |
| --- | ---: | ---: | --- |
| stt | ~459 | ~486 | Flat floor; fixed clip, so variance is not representative of real utterances |
| retrieve | ~8 | ~8 | Negligible |
| generate | ~384 | ~741 | Scales with output length, ~12 ms/token |
| verify | ~2 | ~3 | Negligible |
| tts | ~174 | ~556 | Scales with output length, ~3.5 ms/char |
| **total (warm)** | **~1010** | **~1770** | Within budget |
| **total (cold)** | **~2000–3150** | — | First ask of the longest-answer question exceeds budget |

**Shape.** STT is a constant ~460 ms floor and the largest single stage.
Everything else is driven by answer length: `generate` and `tts` are the same
lever, moving together at roughly 0.75–0.98× of each other. Tightening the
prompt from "2–6 sentences" to "at most 3, stop when answered" cut the median
total from 1309 ms to ~1050 ms by shrinking both stages at once.

**Caveat to state in the demo.** `stt` is measured on one fixed clip, so its
64 ms spread reflects fixed-workload variance, not a distribution over real
utterances.

**Unimplemented improvement.** Sentence-chunked streaming — `llm.stream()` and
`Speaker.split_sentences()` both exist — would make time-to-first-audio
`stt + retrieve + first-sentence generate + first-sentence tts`, roughly 800 ms
and flat regardless of answer length. This is the literal reading of the
handout's "question to the first spoken words" and would remove worst-case
exposure rather than shrinking it.

---

## Open items

- [ ] Score verified text in `evaluate.py` (F7)
- [ ] Regenerate latency figures post-fix at `--runs 20`
- [ ] Revisit the 20-character abstain floor in `verify()` (F1 interaction)
- [ ] Strip inline citation and URL chunks at index time (F5)
- [ ] Decide whether F3 paraphrases should be normalised or remain a defect