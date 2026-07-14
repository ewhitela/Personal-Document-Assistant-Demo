

"""Sanity tests for the Week 7 Transcriber.

    python tests/test_week7_stt.py

The join_segments check runs offline and should reach PASS once you implement it.
The transcription check needs the faster-whisper model and a sample audio file;
it SKIPs cleanly when either is missing, so the file is always safe to run. To
exercise it, drop a short spoken-word clip at tests/data/sample.wav.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.transcriber import join_segments, Transcriber  # noqa: E402


class _Seg:
    """Mimics a faster-whisper segment: it has a .text attribute."""
    def __init__(self, text):
        self.text = text


def test_join_segments():
    segs = [_Seg(" Hello"), _Seg(" there,"), _Seg(" how are you?")]
    out = join_segments(segs)
    assert out == "Hello there, how are you?", f"got {out!r}"
    assert join_segments([]) == "", "empty input should give an empty string"


def test_transcribe_sample():
    t = Transcriber()
    if not t.is_ready():
        print("   note: faster-whisper model not available; skipping.")
        return "skip"
    sample = os.path.join(os.path.dirname(__file__), "data", "sample.wav")
    if not os.path.exists(sample):
        print("   note: no tests/data/sample.wav found; record a short clip to exercise this.")
        return "skip"
    text = t.transcribe(sample)
    assert isinstance(text, str) and text.strip(), "transcribe should return non-empty text"


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
