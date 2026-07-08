"""Sanity tests for the Week 5 LocalLLM.

Run it directly once you have implemented llm.py:

    python tests/test_week5_llm.py

This test needs ollama running with your model pulled. If it is not running,
the tests SKIP rather than fail, so the file is always safe to run. The two
behavior tests are smoke checks: they depend on the model and your prompt, so a
WARN means "look at the output", not necessarily a bug.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.llm import LocalLLM  # noqa: E402


def test_connectivity_and_nonempty():
    llm = LocalLLM()
    if not llm.is_ready():
        print("   note: ollama not reachable or model not pulled. "
              "Start ollama and run `ollama pull <model>`.")
        return "skip"
    out = llm.generate("Reply with exactly one word: pong")
    assert isinstance(out, str) and out.strip(), "generate must return non-empty text"


def test_grounded_answer():
    llm = LocalLLM()
    if not llm.is_ready():
        return "skip"
    context = "Context: The product launch date is March 3, 2026."
    out = llm.generate(context + "\nQuestion: When is the launch?")
    if "march" not in out.lower():
        print("   note: expected the date from the context; got:", repr(out))
        return "warn"


def test_refuses_when_answer_absent():
    llm = LocalLLM()
    if not llm.is_ready():
        return "skip"
    context = "Context: The sky is blue."
    out = llm.generate(context + "\nQuestion: What is the capital of France?").lower()
    refused = any(p in out for p in ("don't know", "do not know", "not in the context",
                                     "no information", "cannot find"))
    if not refused:
        print("   note: the model should refuse when the answer is not in context; got:", repr(out))
        return "warn"


def _run():
    results = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                r = fn()
                status = r.upper() if isinstance(r, str) and r in ("skip", "warn") else "PASS"
                results.append((status, name, ""))
            except NotImplementedError:
                results.append(("TODO", name, "not implemented yet"))
            except AssertionError as e:
                results.append(("FAIL", name, str(e)))
            except Exception as e:  # noqa: BLE001
                results.append(("ERROR", name, repr(e)))
    w = max(len(n) for _, n, _ in results)
    hard = 0
    for s, n, msg in results:
        print(f"{s:5s} {n:<{w}}  {msg}")
        if s in ("FAIL", "ERROR"):
            hard += 1
    print("\n" + ", ".join(f"{s}={sum(1 for x, _, _ in results if x == s)}"
                           for s in ["PASS", "SKIP", "WARN", "TODO", "FAIL", "ERROR"]
                           if any(x == s for x, _, _ in results)))
    raise SystemExit(1 if hard else 0)


if __name__ == "__main__":
    _run()
