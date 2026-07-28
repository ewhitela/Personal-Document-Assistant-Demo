"""
Compare local vs. edge (Jetson) TTS latency for the same text.

Usage:
    python eval/measure_edge_latency.py --edge-url http://jetson.local:5000 \
        --text "This is a latency test." --runs 5

Requires the existing pdva.tts.Speaker for the local leg; imports it
without modification.
"""

import argparse
import statistics
import time

from pdva.tts import Speaker
from pdva.edge_speaker import EdgeSpeaker


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
        print("WARNING: edge service at {} is not ready".format(args.edge_url))

    local_times, edge_times = [], []
    for i in range(args.runs):
        local_times.append(timed(local.synthesize, args.text, "/tmp/local_{}.wav".format(i)))
        edge_times.append(timed(edge.synthesize, args.text, "/tmp/edge_{}.wav".format(i)))

    print("local : mean={:.3f}s  stdev={:.3f}s  runs={}".format(
        statistics.mean(local_times), statistics.stdev(local_times) if args.runs > 1 else 0.0, args.runs))
    print("edge  : mean={:.3f}s  stdev={:.3f}s  runs={}".format(
        statistics.mean(edge_times), statistics.stdev(edge_times) if args.runs > 1 else 0.0, args.runs))
    print("delta : {:+.3f}s (edge - local)".format(statistics.mean(edge_times) - statistics.mean(local_times)))


if __name__ == "__main__":
    main()