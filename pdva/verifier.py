"""Post-generation grounding verification (wraps RAGPipeline unchanged).

v2: adds two checks the first version missed.

1. Citation leakage. SYSTEM_PROMPT forbids in-answer source listings, but the
   3B model sometimes emits "According to passage [2], ..." anyway. v1 treated
   [N] markers as inert and left them in the output. v2 strips the leading
   citation clause (and any bare [N] marker) from the surfaced text. It also
   catches the degraded form where the model drops the passage reference but
   leaves the attribution stub behind ("According to, its sister ship...").

2. Conflation. v1 checked "does this number appear anywhere in the corpus"
   and "does this entity appear anywhere in the corpus" as independent facts,
   so a real number attached to the wrong entity (NAU's answer using the
   Flagstaff Unified School District's enrollment figure) passed silently.
   v2 additionally requires the sentence's number(s) and its named entity
   anchor(s) to co-occur in the SAME source passage. A number that's real but
   sourced from a different passage than the entity it's attached to is
   flagged as "conflated", distinct from "fabricated" (not in the corpus at
   all) -- both still get the sentence stripped, but they're logged
   separately since they're different failure modes for the Week 12 report.

3. Comparison inversion. "X is the Nth-most ADJ NOUN in SCOPE after/before Y"
   claims can have the right entities and still have the direction reversed
   ("Philadelphia is second-most-populous after Pittsburgh" when the source
   says the reverse). Entity-presence and number-co-location checks can't
   catch this since both entities are real and grounded either way -- this
   needs a narrow, pattern-specific check for this one comparison shape.
   It is not general relation extraction and won't catch inverted relations
   phrased differently.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

ABSTAIN = "I don't know based on your documents."

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")

_STATUS_RE = re.compile(
    r"\b(still|today|currently|to this day|remains?|continues?|no longer|"
    r"nowadays|anymore|as of (?:now|today))\b",
    re.IGNORECASE,
)
_STATUS_QUESTION_RE = re.compile(
    r"\b(still (?:exist|stand|true|there|in effect|operat)|today|currently|"
    r"anymore|these days)\b",
    re.IGNORECASE,
)
_STATUS_CLAUSE = "The documents don't state its current status."

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z.\-']*")
_STOP_CAPS = {
    "I",
    "The",
    "A",
    "An",
    "It",
    "This",
    "That",
    "These",
    "Those",
    "However",
    "According",
    "Unlike",
    "Since",
    "There",
    "No",
    "Her",
    "His",
    "Its",
    "Their",
    "In",
    "On",
    "At",
    "As",
    "But",
    "And",
}
_CONNECTORS = {"of", "the", "and", "for", "de", "du"}

# Citation clauses: "According to passage [2], ", "Per source [1] and [3], ".
# Not anchored to sentence-start: "No, according to passage [2], Pittsburgh
# is..." needs the clause stripped even though "No," comes first.
# Second alternative catches the degraded/dangling form -- the model
# sometimes emits "According to, ..." with the passage reference dropped
# but the attribution stub left behind, which the first alternative (which
# requires a passage noun or bracket) doesn't match.
_CITATION_CLAUSE_RE = re.compile(
    r"(?:according to|per|as (?:stated|noted|shown) in)\s+(?:the\s+)?"
    r"(?:passage|source|document)s?\s*"
    r"\[\d+\](?:\s*(?:,|and)\s*\[\d+\])*\s*,?\s*"
    r"|"
    r"(?:according to|per|as (?:stated|noted|shown) in)\s*,\s*",
    re.IGNORECASE,
)
# Anything left over: bare [N] markers anywhere else in the sentence.
_BARE_CITATION_RE = re.compile(r"\s*\[\d+\](?:\s*\[\d+\])*")

# Superlative-comparison pattern: "X is the Nth-most ADJ NOUN in SCOPE
# after/before Y". Regex-based token/entity checks can't tell this apart from
# its factually-inverted twin ("Y is the Nth-most ... after X") since both
# X and Y are real, grounded entities either way -- only the direction is
# wrong. This pattern is narrow by design: it targets this one comparison
# shape rather than attempting general relation extraction.
_ORDINAL_WORDS = r"(?:first|second|third|fourth|fifth|sixth|seventh|\d+(?:st|nd|rd|th))"
# No literal "." in the char class: a trailing period must terminate the
# phrase, not get swallowed as a word char and let matching run into the
# next sentence (caused a false-positive inversion in testing).
_CAP_PHRASE = r"[A-Z][A-Za-z\-']*(?:\s+[A-Za-z\-']+){0,3}"
_COMPARISON_RE = re.compile(
    rf"(?P<subj>{_CAP_PHRASE})\s+is\s+(?:mentioned\s+as\s+|considered\s+)?(?:the\s+)?"
    rf"(?P<ord>{_ORDINAL_WORDS})[\s-]*most\s+(?P<adj>[a-z]+)\s+(?P<noun>[a-z]+)\s+in\s+"
    rf"(?P<scope>{_CAP_PHRASE})\s+(?P<rel>after|before)\s+(?P<obj>{_CAP_PHRASE})",
)
_ORDINAL_ALIASES = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
}

def _key(word: str) -> str:
    k = word.rstrip(".-'").lower()
    return k.removesuffix("'s")

def _norm_ordinal(o: str) -> str:
    return _ORDINAL_ALIASES.get(o.lower(), o.lower())


def _check_comparison_inversions(sentence: str, passage_texts: list[str]) -> list[str]:
    """Return human-readable descriptions of superlative claims whose
    subject/object direction contradicts what a passage actually says."""
    problems = []
    for m in _COMPARISON_RE.finditer(sentence):
        subj, obj = m.group("subj").strip(), m.group("obj").strip().rstrip(".,;: ")
        ordw = _norm_ordinal(m.group("ord"))
        adj, noun = re.escape(m.group("adj")), re.escape(m.group("noun"))
        scope, rel = re.escape(m.group("scope").strip()), m.group("rel")
        tail_re = re.compile(
            rf"{ordw}[\s-]*most\s+{adj}\s+{noun}\s+in\s+{scope}\s+{rel}\s+"
            rf"(?P<pobj>{_CAP_PHRASE})",
            re.IGNORECASE,
        )
        for text in passage_texts:
            pm = tail_re.search(text)
            if not pm:
                continue
            pobj = pm.group("pobj").strip().rstrip(".,;: ")
            if pobj.split()[0].lower() != obj.split()[0].lower():
                problems.append(
                    f'"{subj} ... {rel} {obj}" reverses the source, which says '
                    f'"... {rel} {pobj}"'
                )
            break
    return problems


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _cap_sequences(text: str) -> list[list[str]]:
    """Runs of capitalized words (lowercase connectors allowed mid-run,
    excluded from initials), for acronym matching: "University of Pittsburgh
    Medical Center" -> UPMC."""
    seqs, cur = [], []
    for w in _words(text):
        if w[:1].isupper():
            cur.append(w)
        elif w in _CONNECTORS and cur:
            continue
        else:
            if len(cur) >= 2:
                seqs.append(cur)
            cur = []
    if len(cur) >= 2:
        seqs.append(cur)
    return seqs


def _norm(num: str) -> str:
    return num.replace(",", "").rstrip(".")


def _extract_numbers(text: str) -> set[str]:
    return {_norm(m.group()) for m in _NUMBER_RE.finditer(text)}


def _strip_citations(sentence: str) -> tuple[str, list[str]]:
    """Remove in-answer citation markers. Returns (clean_sentence, removed)."""
    removed = []
    for m in _CITATION_CLAUSE_RE.finditer(sentence):
        brackets = re.findall(r"\[\d+\]", m.group())
        removed.extend(brackets if brackets else ["according to (dangling)"])
    s = _CITATION_CLAUSE_RE.sub("", sentence)
    s = _BARE_CITATION_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s, removed


def _entity_anchors(sentence: str) -> list[str]:
    """Capitalized-word candidates in a sentence that could anchor a number
    to a specific subject (skips the sentence-initial word, which may just
    be capitalized for grammar)."""
    words = _words(sentence)
    return [
        w for w in words[1:] if w[:1].isupper() and w not in _STOP_CAPS and len(w) >= 3
    ]


def _anchors_present(
    anchors: list[str], words_set: set[str], initials_set: set[str]
) -> bool:
    """All anchors must be supported by this one passage."""
    for a in anchors:
        key = _key(a)
        if key in words_set:
            continue
        if a.isupper() and a in initials_set:
            continue
        return False
    return True


def _ungrounded_entities(
    sentence: str, grounding_lower_words: set[str], corpus_initials: set[str]
) -> list[str]:
    bad = []
    for w in _entity_anchors(sentence):
        key = _key(w)
        if key in grounding_lower_words:
            continue
        if w.isupper() and w in corpus_initials:
            continue
        bad.append(w)
    return bad


@dataclass
class Verification:
    """What the check did, for eval logs and the Week 12 report."""

    passed: bool  # True = output identical to input
    abstained: bool = False  # True = whole answer replaced
    ungrounded: list[str] = field(default_factory=list)  # fabricated numbers
    conflated: list[str] = field(default_factory=list)  # real numbers, wrong entity
    stripped: list[str] = field(default_factory=list)  # removed sentences
    ungrounded_entities: list[str] = field(default_factory=list)
    status_stripped: bool = False
    citations_stripped: list[str] = field(default_factory=list)  # e.g. ["[2]"]
    inversions: list[str] = field(default_factory=list)  # reversed comparisons


def verify(
    answer: str, passages: Iterable, question: str = ""
) -> tuple[str, Verification]:
    if answer.strip() == ABSTAIN:
        return answer, Verification(passed=True, abstained=True)

    passage_texts = [p.text for p in passages]
    passage_nums = [_extract_numbers(t) for t in passage_texts]
    passage_words = [
        {_key(w) for w in _words(t)} for t in passage_texts
    ]
    passage_initials = [
        {"".join(w[0].upper() for w in seq) for seq in _cap_sequences(t)}
        for t in passage_texts
    ]

    context = " ".join(passage_texts)
    grounding = context + " " + question
    grounding_words = {_key(w) for w in _words(grounding)}
    corpus_initials = {
        "".join(w[0].upper() for w in seq) for seq in _cap_sequences(grounding)
    }
    context_has_status = bool(_STATUS_RE.search(context))
    status_question = bool(_STATUS_QUESTION_RE.search(question))

    kept, rep = [], Verification(passed=True)

    for raw_sent in _SENT_RE.split(answer.strip()):
        sent, citation_markers = _strip_citations(raw_sent)
        if citation_markers:
            rep.citations_stripped.extend(citation_markers)

        anchors = _entity_anchors(sent)
        bad_ents = _ungrounded_entities(sent, grounding_words, corpus_initials)

        fabricated, conflated = [], []
        for num in _extract_numbers(sent):
            occurs_anywhere = any(num in nums for nums in passage_nums)
            if not occurs_anywhere:
                fabricated.append(num)
                continue
            if not anchors:
                continue  # nothing to attribute to; presence is all we can check
            co_located = any(
                num in passage_nums[i]
                and _anchors_present(anchors, passage_words[i], passage_initials[i])
                for i in range(len(passage_texts))
            )
            if not co_located:
                conflated.append(num)

        bad_status = bool(_STATUS_RE.search(sent)) and not context_has_status
        bad_inversions = _check_comparison_inversions(sent, passage_texts)

        if fabricated or conflated or bad_ents or bad_status or bad_inversions:
            rep.stripped.append(raw_sent)
            rep.ungrounded.extend(fabricated)
            rep.conflated.extend(conflated)
            rep.ungrounded_entities.extend(bad_ents)
            rep.status_stripped = rep.status_stripped or bad_status
            rep.inversions.extend(bad_inversions)
        else:
            kept.append(sent)

    if not rep.stripped and not rep.citations_stripped:
        return answer, rep

    remainder = " ".join(kept).strip()
    if len(remainder) < 20:
        rep.abstained = True
        text = ABSTAIN
    else:
        text = remainder

    if status_question and rep.status_stripped and text != ABSTAIN:
        text = text.rstrip() + " " + _STATUS_CLAUSE

    rep.passed = text == answer.strip()
    return text, rep


class VerifiedRAGPipeline:
    """Drop-in wrapper: same .answer() signature, verified output."""

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def answer(self, question: str):
        result = self.pipeline.answer(question)
        text, report = verify(result.answer, result.sources, question)
        result.answer = text
        result.verification = report
        return result
