

"""Sanity tests for the Week 8 Speaker.

    python tests/test_week8_tts.py

The split_sentences check runs offline and should reach PASS once implemented.
The synthesis check needs Piper and a downloaded voice; it SKIPs cleanly when
either is missing, and otherwise writes a short WAV and verifies it is real audio.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.tts import split_sentences, Speaker  # noqa: E402


def test_split_sentences():
    out = split_sentences("Hello world. How are you? I am fine!")
    assert out == ["Hello world.", "How are you?", "I am fine!"], f"got {out!r}"
    assert split_sentences("   ") == [], "whitespace-only input should give an empty list"


def test_synthesize_wav():
    sp = Speaker()
    if not sp.is_ready():
        print("   note: Piper or the voice file is not available; skipping.")
        return "skip"
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "out.wav")
    path = sp.synthesize("This is a short test.", out)
    assert os.path.exists(path) and os.path.getsize(path) > 44, "synthesize should write a non-trivial WAV"
    with open(path, "rb") as f:
        head = f.read(12)
    assert head[:4] == b"RIFF" and head[8:12] == b"WAVE", "output should be a valid WAV file"


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

