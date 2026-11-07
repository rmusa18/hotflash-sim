"""Illustrative software-only pulse train for virtual stimulation."""

from math import ceil, floor, isfinite

import numpy as np
from numpy.typing import NDArray

from .vns_controller import StimulationState, VNSController


def generate_biphasic_waveform(
    duration_s: float,
    *,
    frequency_hz: float = 25.0,
    pulse_width_ms: float = 0.2,
    amplitude: float = 1.0,
    interphase_ms: float = 0.0,
    sample_rate_hz: int = 50_000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return time in seconds from pulse-train onset and amplitude in arbitrary units.

    Pulse starts, phase widths and the optional gap are rounded to the sample
    grid. Each pulse has equal positive and negative phases. A pulse is omitted
    if both phases cannot fit within the requested duration.
    """
    positive = (frequency_hz, pulse_width_ms, amplitude, sample_rate_hz)
    nonnegative = (duration_s, interphase_ms)
    if not all(isfinite(value) and value > 0 for value in positive):
        raise ValueError("Frequency, pulse width, amplitude and sample rate must be positive and finite.")
    if not all(isfinite(value) and value >= 0 for value in nonnegative):
        raise ValueError("Duration and interphase gap must be nonnegative and finite.")

    samples_per_period = sample_rate_hz / frequency_hz
    phase_samples = round(pulse_width_ms * sample_rate_hz / 1000)
    gap_samples = round(interphase_ms * sample_rate_hz / 1000)
    if phase_samples < 1:
        raise ValueError("Sample rate is too low to resolve the requested pulse width.")
    pulse_samples = 2 * phase_samples + gap_samples
    if pulse_samples > floor(samples_per_period):
        raise ValueError("Both phases and the gap must fit within one pulse period.")

    sample_count = floor(duration_s * sample_rate_hz)
    time_s = np.arange(sample_count, dtype=float) / sample_rate_hz
    waveform = np.zeros(sample_count, dtype=float)

    pulse_numbers = np.arange(ceil(sample_count / samples_per_period))
    starts = np.rint(pulse_numbers * samples_per_period).astype(np.intp)
    starts = starts[starts + pulse_samples <= sample_count]
    positive_phase = starts[:, None] + np.arange(phase_samples)
    waveform[positive_phase] = amplitude
    waveform[positive_phase + phase_samples + gap_samples] = -amplitude
    return time_s, waveform


def active_waveform_preview(
    controller: VNSController,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the first 200 ms of the active train for display, timed from its onset.

    For shorter active windows the preview is clipped to the scheduled duration.
    Monitoring and cooldown return two empty arrays.
    """
    if controller.state is not StimulationState.STIMULATION_ACTIVE:
        return np.empty(0), np.empty(0)
    stimulation = controller.stimulations[-1]
    return generate_biphasic_waveform(min(0.2, stimulation.end_s - stimulation.start_s))
