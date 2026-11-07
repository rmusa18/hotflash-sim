"""Offline comparisons of explicitly assumed responses to virtual stimulation."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .vns_controller import Stimulation


@dataclass(frozen=True)
class ResponseScenario:
    signals: pd.DataFrame
    stimulations: tuple[Stimulation, ...]
    assumed_effect_fraction: float


def _response_envelope(
    time_s: NDArray[np.float64],
    stimulation: Stimulation,
    delay_s: float,
    development_s: float,
    recovery_s: float,
) -> NDArray[np.float64]:
    onset_s = stimulation.start_s + delay_s
    offset_s = stimulation.end_s + delay_s
    if development_s > 0:
        rise = np.clip((time_s - onset_s) / development_s, 0.0, 1.0)
        peak = min((offset_s - onset_s) / development_s, 1.0)
    else:
        rise = (time_s >= onset_s).astype(float)
        peak = 1.0

    if recovery_s > 0:
        tail = peak * np.clip(1.0 - (time_s - offset_s) / recovery_s, 0.0, 1.0)
    else:
        tail = 0.0
    return np.where(time_s < offset_s, rise, tail)


def simulate_response_scenarios(
    signals: pd.DataFrame,
    stimulations: Iterable[Stimulation],
    *,
    baseline_eda_uS: ArrayLike,
    assumed_effect_fraction: float,
    response_delay_s: float = 0.0,
    development_s: float = 0.0,
    recovery_s: float = 0.0,
) -> dict[str, ResponseScenario]:
    """Compare three scenarios from the same synthetic EDA, pulse and temperature.

    The baseline is a scalar or one value per frame. Only EDA above the
    baseline can be attenuated. The assumed fraction is between zero and one.
    The response ramps up linearly after start plus delay and decays linearly
    after end plus delay. Overlapping responses use the strongest envelope.
    Zero development or recovery time produces an immediate transition.
    """
    if not 0 <= assumed_effect_fraction <= 1:
        raise ValueError("Assumed effect fraction must be between 0 and 1.")
    durations = (response_delay_s, development_s, recovery_s)
    if not all(isfinite(value) and value >= 0 for value in durations):
        raise ValueError("Response delay, development and recovery must be finite nonnegative seconds.")

    columns = ["t_s", "eda_uS", "pr_bpm", "skin_temp_C"]
    original = signals.loc[:, columns]
    time_s = original["t_s"].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(time_s).all() or (time_s < 0).any() or (np.diff(time_s) <= 0).any():
        raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing.")
    baseline = np.asarray(baseline_eda_uS, dtype=float)
    if (baseline.ndim != 0 and baseline.shape != time_s.shape) or not np.isfinite(baseline).all():
        raise ValueError("EDA baseline must be a finite scalar or one finite value per frame.")

    windows = tuple(stimulations)
    assumed = original.copy()
    if assumed_effect_fraction > 0:
        envelope = np.zeros_like(time_s)
        for window in windows:
            if window.end_s > window.start_s:
                response = _response_envelope(time_s, window, *durations)
                np.maximum(envelope, response, out=envelope)

        eda = original["eda_uS"].to_numpy(dtype=float, na_value=np.nan, copy=True)
        excursion = eda - baseline
        affected = (excursion > 0) & (envelope > 0)
        if affected.any():
            eda[affected] -= assumed_effect_fraction * envelope[affected] * excursion[affected]
            assumed["eda_uS"] = eda

    return {
        "no_stimulation": ResponseScenario(original.copy(), (), 0.0),
        "zero_effect": ResponseScenario(original.copy(), windows, 0.0),
        "assumed_effect": ResponseScenario(assumed, windows, assumed_effect_fraction),
    }
