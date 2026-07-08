"""Sanity tests for the Week 4 DocumentIndex.

Run it directly once you have implemented embedding_index.py:

    python tests/test_week4_index.py

Before implementing, every test shows TODO. After implementing, you want all
PASS. This test indexes a few tiny text files, so the first run downloads the
embedding model (a one-time, ~80 MB download).
"""
import os
import sys
import shutil
import tempfile

# Make `import pdva` work no matter which directory you run this from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.embedding_index import DocumentIndex  # noqa: E402

DOCS = {
    "passwords.txt": "To reset your password, open Settings, choose Security, then Reset password.",
    "refunds.txt": "Our refund policy allows returns within 30 days of purchase with a receipt.",
    "hours.txt": "The office is open Monday to Friday, nine in the morning to five in the evening.",
}


def _write_docs(folder):
    paths = ['/home/ra/Downloads/Nova Ircutia - Starlit Isles Wiki.pdf','/home/ra/Downloads/Olympia - Starlit Isles Wiki.pdf']
    for name, text in DOCS.items():
        p = os.path.join(folder, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        paths.append(p)
    return paths


def test_index_and_search():
    tmp = tempfile.mkdtemp()
    try:
        idx = DocumentIndex(persist_dir=os.path.join(tmp, "store"))
        idx.reset()
        added = idx.add_documents(_write_docs(tmp))
        assert added > 0, "add_documents should return the number of chunks added"
        assert idx.count() > 0, "count should be > 0 after indexing"

        hits = idx.search("how do I change my password?", k=2)
        assert len(hits) >= 1, "search should return at least one passage"
        assert hits[0].source == "passwords.txt", \
            f"top hit should be passwords.txt, got {hits[0].source!r}"
        assert 0.0 <= hits[0].score <= 1.0, "score should be a 0..1 similarity"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_persistence():
    tmp = tempfile.mkdtemp()
    try:
        store = os.path.join(tmp, "store")
        idx = DocumentIndex(persist_dir=store)
        idx.reset()
        idx.add_documents(_write_docs(tmp))
        before = idx.count()

        del idx
        reopened = DocumentIndex(persist_dir=store)  # new process would do the same
        assert reopened.count() == before, "the index must persist across restarts"
        assert len(reopened.search("norderlands", k=1)) >= 1, \
            "search must work after reopening without re-indexing"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_idempotent_reindex():
    tmp = tempfile.mkdtemp()
    try:
        idx = DocumentIndex(persist_dir=os.path.join(tmp, "store"))
        idx.reset()
        paths = _write_docs(tmp)
        idx.add_documents(paths)
        first = idx.count()
        idx.add_documents(paths)  # exactly the same documents again
        second = idx.count()
        assert first == second, \
            f"re-indexing the same documents must not grow the index ({first} -> {second})"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
