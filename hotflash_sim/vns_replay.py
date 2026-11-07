"""Build a dashboard replay from observable frames and a detector callback.

Each frame's active_stimulation_index points to the corresponding entry in
stimulations, or is null when stimulation is off. Each response scenario's
EDA array aligns with frames by index and shares the original pulse rate and
temperature. Response curves are computed after the detector finishes.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from math import isfinite
from numbers import Real
from typing import TypedDict

import pandas as pd

from .response_model import simulate_response_scenarios
from .vns_controller import StimulationState, VNSController
from .vns_waveform import active_waveform_preview


class ObservableFrame(TypedDict):
    t_s: float
    eda_uS: float | None
    pr_bpm: float | None
    skin_temp_C: float | None


def _finite_number(value: object) -> float | None:
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if isfinite(number) else None


def build_dashboard_replay(
    frames: Iterable[Mapping[str, object]],
    detect: Callable[[ObservableFrame], Mapping[str, object] | None],
    *,
    detection_source: str,
    stimulation_s: float,
    cooldown_s: float,
    baseline_eda_uS: float,
    assumed_effect_fraction: float,
    response_delay_s: float = 0.0,
    development_s: float = 0.0,
    recovery_s: float = 0.0,
) -> dict[str, object]:
    """Call detect once per observable frame and return a JSON-ready replay."""
    controller = VNSController(stimulation_s=stimulation_s, cooldown_s=cooldown_s)
    timeline: list[dict[str, object]] = []
    stimulations: list[dict[str, object]] = []
    previous_time_s = -1.0

    for raw_frame in frames:
        time_s = _finite_number(raw_frame.get("t_s"))
        if time_s is None or time_s < 0 or time_s <= previous_time_s:
            raise ValueError("Replay timestamps must be finite, nonnegative and strictly increasing.")
        previous_time_s = time_s
        frame: ObservableFrame = {
            "t_s": time_s,
            "eda_uS": _finite_number(raw_frame.get("eda_uS")),
            "pr_bpm": _finite_number(raw_frame.get("pr_bpm")),
            "skin_temp_C": _finite_number(raw_frame.get("skin_temp_C")),
        }
        controller.update(time_s)
        detection = detect(frame.copy())
        if detection is not None:
            if isinstance(detection, Mapping):
                detected_at_s = _finite_number(detection.get("detected_at_s"))
                if detected_at_s is not None and detected_at_s > time_s:
                    # Mark a look-ahead message invalid without advancing the replay clock.
                    detection = {**detection, "detected": False}
            stimulation = controller.on_detection(detection)
            if stimulation is not None:
                pulse_time_s, amplitude = active_waveform_preview(controller)
                stimulations.append({
                    **asdict(stimulation),
                    "duration_s": controller.stimulation_s,
                    "cooldown_s": controller.cooldown_s,
                    "waveform_preview": {
                        "time_from_start_s": pulse_time_s.tolist(),
                        "amplitude_au": amplitude.tolist(),
                    },
                })

        active = controller.state is StimulationState.STIMULATION_ACTIVE
        timeline.append({
            **frame,
            "controller_state": controller.state.value,
            "active_stimulation_index": len(stimulations) - 1 if active else None,
        })

    if not timeline:
        raise ValueError("The replay needs at least one observable frame.")
    scenarios = simulate_response_scenarios(
        pd.DataFrame(timeline), controller.stimulations,
        baseline_eda_uS=baseline_eda_uS, assumed_effect_fraction=assumed_effect_fraction,
        response_delay_s=response_delay_s, development_s=development_s, recovery_s=recovery_s,
    )
    scenario_labels = {
        "no_stimulation": "No stimulation",
        "zero_effect": "Virtual stimulation with zero modelled effect",
        "assumed_effect": "Virtual stimulation with an assumed EDA effect",
    }
    return {
        "schema_version": 1,
        "signal_source": "synthetic",
        "detection_source": detection_source,
        "stimulation_mode": "virtual_taVNS",
        "human_stimulation": False,
        "display_labels": {
            "signals": "Synthetic physiological signals",
            "stimulation": "Virtual auricular vagus nerve stimulation",
            "waveform": "Illustrative biphasic waveform (arbitrary units)",
            "assumed_effect_fraction": "Assumed reduction of EDA above baseline",
        },
        "frames": timeline,
        "stimulations": stimulations,
        "trigger_log": [asdict(decision) for decision in controller.trigger_log],
        "response_settings": {
            "baseline_eda_uS": float(baseline_eda_uS),
            "response_delay_s": float(response_delay_s),
            "development_s": float(development_s),
            "recovery_s": float(recovery_s),
        },
        "response_scenarios": {
            name: {
                "label": scenario_labels[name],
                "signal_source": "hypothetical" if name == "assumed_effect" else "synthetic",
                "stimulation_enabled": name != "no_stimulation",
                "assumed_effect_fraction": float(scenario.assumed_effect_fraction),
                "eda_uS": [_finite_number(eda) for eda in scenario.signals["eda_uS"]],
            }
            for name, scenario in scenarios.items()
        },
    }
