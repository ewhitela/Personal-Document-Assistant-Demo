

"""Sanity tests for the Week 9 vision module (optional).

    python tests/test_week9_vision.py

Offline checks (should reach PASS after you implement the helpers, or PASS now
for the routing check, since the VisionModel facade is provided):
    - build_image_message   builds the ollama message
    - encode_image_b64      base64-encodes an image file
    - backend routing       VisionModel delegates to whatever backend it is given
The live check needs a real backend (ollama vision model, or your remote server)
and a sample image at tests/data/sample.png; it SKIPs otherwise.
"""
import base64
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.vision import (  # noqa: E402
    build_image_message, encode_image_b64, VisionModel, VisionBackend,
)


def test_build_image_message():
    msg = build_image_message("/tmp/pic.png", "What is in this image?")
    assert msg.get("role") == "user", "role should be 'user'"
    assert msg.get("content") == "What is in this image?", "content should be the question"
    assert msg.get("images") == ["/tmp/pic.png"], "images should be a list holding the path"


def test_encode_image_b64():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "tiny.bin")
    raw = b"\x89PNG\r\n\x1a\n_hello_bytes_"
    with open(path, "wb") as f:
        f.write(raw)
    out = encode_image_b64(path)
    assert isinstance(out, str), "should return a str"
    assert base64.b64decode(out) == raw, "decoding the result should recover the original bytes"


class _FakeBackend(VisionBackend):
    """A backend that needs no model, to test that the facade delegates to it."""
    def is_ready(self):
        return True
    def ask(self, image_path, question):
        return f"seen:{os.path.basename(image_path)}|q:{question}"


def test_backend_routing():
    vm = VisionModel(backend=_FakeBackend())
    assert vm.is_ready() is True, "facade should delegate is_ready to the backend"
    out = vm.ask("/tmp/cat.png", "what animal?")
    assert out == "seen:cat.png|q:what animal?", "facade.ask should call backend.ask"
    desc = vm.describe("/tmp/cat.png")
    assert desc.startswith("seen:cat.png|q:"), "describe should route through ask with a fixed prompt"


def test_live_ask():
    vm = VisionModel()  # default local backend
    if not vm.is_ready():
        print("   note: no local vision model available; skipping live check.")
        return "skip"
    sample = os.path.join(os.path.dirname(__file__), "data", "sample.png")
    if not os.path.exists(sample):
        print("   note: no tests/data/sample.png found; add one to exercise this.")
        return "skip"
    out = vm.ask(sample, "Describe this image in one sentence.")

    print(f"Response: {out}")

    assert isinstance(out, str) and out.strip(), "ask should return non-empty text"


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