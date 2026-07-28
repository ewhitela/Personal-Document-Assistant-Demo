"""
Compare local vs. edge (Jetson) TTS latency for the same text.

Usage:
    python eval/edge_latency.py --edge-url http://jetson.local:5000 \
        --text "This is a latency test." --runs 5

Requires the existing pdva.tts.Speaker for the local leg; imports it
without modification.
"""

import argparse
import statistics
import time

from pdva.edge_speaker import EdgeSpeaker
from pdva.tts import Speaker


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-url", required=True)
    ap.add_argument("--text", default="This is a latency test.")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    local = Speaker()
    edge = EdgeSpeaker(args.edge_url)

    if not edge.is_ready():
        print(f"WARNING: edge service at {args.edge_url} is not ready")

    local_times, edge_times = [], []
    for i in range(args.runs):
        local_times.append(timed(local.synthesize, args.text, f"/tmp/local_{i}.wav"))
        edge_times.append(timed(edge.synthesize, args.text, f"/tmp/edge_{i}.wav"))

    print(
        f"local : mean={statistics.mean(local_times):.3f}s  stdev={statistics.stdev(local_times) if args.runs > 1 else 0.0:.3f}s  runs={args.runs}"
    )
    print(
        f"edge  : mean={statistics.mean(edge_times):.3f}s  stdev={statistics.stdev(edge_times) if args.runs > 1 else 0.0:.3f}s  runs={args.runs}"
    )
    print(
        f"delta : {statistics.mean(edge_times) - statistics.mean(local_times):+.3f}s (edge - local)"
    )


if __name__ == "__main__":
    main()
