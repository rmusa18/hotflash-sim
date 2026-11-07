"""Example: consume the real-time stream (one JSON frame every 500 ms).

Either pipe the CLI:
    hotflash-sim stream --seed 42 --speed 60 | python examples/stream_consumer.py

or run it standalone, which uses the Python API:
    python examples/stream_consumer.py --standalone --speed 60
"""
import argparse
import json
import sys
import time


def handle(frame, state):
    """Your online logic goes here. This one prints a line per minute plus alerts."""
    if frame["t_s"] % 60 < 0.5:
        print(f"t={frame['t_s'] / 60:6.1f} min  EDA={frame['eda_uS']}  skin={frame['skin_temp_C']}  "
              f"PR={frame['pr_bpm']}  RMSSD={frame['prv_rmssd_60s_ms']}", flush=True)
    for a in frame.get("alerts", []):
        print("ALERT", a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standalone", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--speed", type=float, default=60)
    a = ap.parse_args()
    state = {}
    if a.standalone:
        from hotflash_sim import RealtimeStream, OnlineFreedmanDetector
        s = RealtimeStream(seed=a.seed)
        det = OnlineFreedmanDetector(1 / s.period)
        t_next = time.monotonic()
        for f in s.frames():
            f["alerts"] = det.update(f)
            handle(f, state)
            t_next += s.period / a.speed
            time.sleep(max(0.0, t_next - time.monotonic()))
    else:
        for line in sys.stdin:
            handle(json.loads(line), state)


if __name__ == "__main__":
    main()
