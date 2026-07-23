"""Scratch diagnostic — not part of the pipeline.

Inspects where the elevation-bearing chunk ranks for a query that keeps
failing to retrieve it. Run from the project root with the venv active:

    python diagnose_retrieval.py

Requires the index to already be built (pdva_store/ populated from a prior
add_documents() call) so this just opens it read-only — it does not re-index.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.embedding_index import DocumentIndex

QUERIES = [
    "What is Flagstaff's elevation?",
    "elevation of Flagstaff",
    "How high is Flagstaff above sea level?",
    "What's the elevation of Flagstaff, and why does that matter for the city's climate?",
]

K = 15


def main() -> None:
    index = DocumentIndex()  # loads existing PERSIST_DIR / COLLECTION_NAME per config.py

    print(f"Indexed chunks: {index.count()}\n")

    for query in QUERIES:
        print(f"=== query: {query!r} ===")
        results = index.search(query, K)

        found_elevation = False
        for i, p in enumerate(results, 1):
            snippet = p.text[:150].replace("\n", " ")
            flag = ""
            if "elevation" in p.text.lower() or "feet" in p.text.lower():
                flag = "  <-- mentions elevation/feet"
                if "flagstaff lies at" in p.text.lower() or "elevation" in p.text.lower():
                    found_elevation = True
            print(f"{i:2d}  score={p.score:.4f}  source={p.source}  {snippet}{flag}")

        if not found_elevation:
            print(f"  ** elevation-bearing chunk NOT in top {K} for this phrasing **")
        print()


if __name__ == "__main__":
    main()