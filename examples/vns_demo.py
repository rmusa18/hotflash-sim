"""Write a dashboard replay. Run with python -m examples.vns_demo."""

import argparse
import json
from functools import partial
from itertools import takewhile
from pathlib import Path
from statistics import median
from typing import Literal

from hotflash_sim import OnlineFreedmanDetector, RealtimeStream
from hotflash_sim.vns_replay import ObservableFrame, build_dashboard_replay


MOCK_DETECTION_S = 2329.75


def mock_detection(frame: ObservableFrame) -> dict[str, object] | None:
    if frame["t_s"] != MOCK_DETECTION_S:
        return None
    return {
        "session_id": "seed42-demo",
        "event_id": "MOCK01",
        "detected_at_s": frame["t_s"],
        "detected": True,
    }


def baseline_detection(
    detector: OnlineFreedmanDetector, frame: ObservableFrame,
) -> dict[str, object] | None:
    """Adapt Nathan's first EDA alert for each episode to the controller event format."""
    for alert in detector.update(frame):
        if alert["alert"] == "eda_rise":
            time_s = alert["t_s"]
            return {
                "session_id": "seed42-demo",
                "event_id": f"EDA-{time_s}",
                "detected_at_s": time_s,
                "detected": True,
            }
    return None


def build_demo_replay(detector_mode: Literal["mock", "baseline"]) -> dict[str, object]:
    """Generate the seeded example used by the CLI and end-to-end test."""
    stream = RealtimeStream(seed=42)
    if detector_mode == "baseline":
        detect = partial(baseline_detection, OnlineFreedmanDetector(rate_hz=1 / stream.period))
        detection_source = "nathan_online_eda_baseline"
    else:
        detect = mock_detection
        detection_source = "mock"

    end_s = MOCK_DETECTION_S + 100
    frames = [frame for frame in takewhile(lambda frame: frame["t_s"] <= end_s, stream.frames())
              if frame["t_s"] >= MOCK_DETECTION_S - 65]
    baseline = median(frame["eda_uS"] for frame in frames
                      if frame["t_s"] <= MOCK_DETECTION_S - 35 and frame["eda_uS"] is not None)
    return build_dashboard_replay(
        (frame for frame in frames if frame["t_s"] >= MOCK_DETECTION_S - 40), detect,
        detection_source=detection_source, stimulation_s=30, cooldown_s=60, baseline_eda_uS=baseline,
        assumed_effect_fraction=0.30, response_delay_s=4, development_s=8, recovery_s=35,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("mock", "baseline"), default="mock",
                        help="Use a fixed mock event or Nathan's online EDA baseline.")
    parser.add_argument("--out", type=Path, help="Override the output JSON path.")
    args = parser.parse_args()

    replay = build_demo_replay(args.detector)
    output = json.dumps(replay, allow_nan=False, separators=(",", ":"))
    default_out = Path("outputs/vns_baseline_replay.json" if args.detector == "baseline" else "outputs/vns_replay.json")
    out = args.out if args.out is not None else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(output + "\n", encoding="utf-8")
    print(f"Wrote dashboard replay to {out}")


if __name__ == "__main__":
    main()
