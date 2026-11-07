"""Negative examples (physiological mimics) and measurement artifacts.

Both layers produce labelled events so a detector can be scored on false positives.

Negatives (physiology, before beat generation):
  arousal        brief awakening: fast HR jump, vagal dip, small fast SCR, maybe movement
  sc_storm       EDA storm in slow-wave sleep: NS-SCR burst, no HR change
  ambient_heat   warm room/bedding: slow sustained SC rise, mild HR rise
  plm_series     periodic limb movements: movement every ~30 s with small HR/SC responses

Artifacts (measurement layer, after physiology):
  eda_motion, device_contact_loss (EDA + PPG + skin temp), ppg_missed, ppg_extra,
  ppg_motion (movement-corrupted pulses), ppg_dropout, ectopic (premature beats),
  plus EDA quantisation and saturation.

All rates and amplitudes are ASSUMED placeholders (see params.py).
"""
from __future__ import annotations

import numpy as np


def _k(t, tr, td):
    from .simulate import _kernel
    return _kernel(t, tr, td)


def _add(arr, t, t0, amp, tr, td, span=8.0):
    dt = t[1] - t[0]
    i0 = max(int(t0 / dt), 0)
    i1 = min(i0 + int(span * td / dt) + 1, len(t))
    if i0 < len(t):
        arr[i0:i1] += amp * _k(t[i0:i1] - t0, tr, td)


def _poisson_times(rng, rate_per_h, T, margin=60.0):
    margin = min(margin, T / 4)  # short simulations
    n = rng.poisson(rate_per_h * T / 3600)
    return np.sort(rng.uniform(margin, T - margin, n))


def generate_negatives(t, v, rng):
    """Return additive components and labels for physiological mimics."""
    T = t[-1] + (t[1] - t[0])
    n = len(t)
    s = v["negative_rate_scale"]
    out = {k: np.zeros(n) for k in ("dhr", "dvagal", "dsc", "dresp", "acc",
                                    "dvasocon", "dvasodil", "dskin")}
    labels = []

    for t0 in _poisson_times(rng, s * v["arousal_rate"], T):
        dhr = max(rng.normal(v["arousal_dhr"], 3.0), 1.0)
        amp = v["arousal_sc_median"] * 2.0 ** rng.normal()
        _add(out["dhr"], t, t0, dhr, 2.0, 15.0)
        _add(out["dvagal"], t, t0, -0.2, 2.0, 15.0)
        _add(out["dsc"], t, t0 + 1.5, amp, 1.5, 8.0)
        _add(out["dresp"], t, t0, 2.0, 2.0, 20.0)
        _add(out["dvasocon"], t, t0 + 2, v["ppg_vasocon_arousal"] * rng.uniform(0.5, 1.5), 3.0, 20.0)
        _add(out["dskin"], t, t0 + 10, -rng.uniform(0.05, 0.25), 30.0, 180.0)
        moved = rng.random() < v["arousal_move_prob"]
        if moved:
            _add(out["acc"], t, t0, rng.uniform(0.05, 0.3), 0.3, 2.0)
        labels.append(dict(type="arousal", category="negative", onset_s=t0,
                           offset_s=t0 + 60, amplitude=amp, dhr_bpm=dhr, moved=moved))

    for t0 in _poisson_times(rng, s * v["storm_per_night"] / 8, T, 300):
        dur = max(rng.exponential(v["storm_duration"]), 5.0) * 60
        k = rng.poisson(v["storm_scr_rate"] * dur / 60)
        for tt in rng.uniform(t0, t0 + dur, k):
            _add(out["dsc"], t, tt, v["storm_scr_amp"] * 1.6 ** rng.normal(), 1.0, 4.0)
        labels.append(dict(type="sc_storm", category="negative", onset_s=t0,
                           offset_s=t0 + dur, amplitude=v["storm_scr_amp"], n_scr=k))

    for t0 in _poisson_times(rng, s * v["heat_per_night"] / 8, T, 600):
        amp = v["heat_sc_amp"] * 1.5 ** rng.normal()
        dur = rng.uniform(30, 60) * 60
        ramp = np.clip((t - t0) / 600.0, 0, 1) * np.clip((t0 + dur - t) / 900.0 + 1, 0, 1)
        out["dsc"] += amp * ramp
        out["dhr"] += v["heat_dhr"] * ramp
        out["dvasodil"] += 0.3 * ramp
        out["dskin"] += rng.uniform(0.4, 1.0) * ramp
        labels.append(dict(type="ambient_heat", category="negative", onset_s=t0,
                           offset_s=t0 + dur + 900, amplitude=amp))

    for t0 in _poisson_times(rng, s * v["plm_series_per_night"] / 8, T, 300):
        dur = rng.uniform(5, 30) * 60
        tt = t0
        while tt < min(t0 + dur, T - 30):
            _add(out["acc"], t, tt, rng.uniform(0.02, 0.1), 0.3, 1.5)
            _add(out["dhr"], t, tt + 2, rng.uniform(2, 6), 2.0, 8.0)
            _add(out["dsc"], t, tt + 2, 0.15 * 1.8 ** rng.normal(), 1.0, 4.0)
            _add(out["dvasocon"], t, tt + 2, rng.uniform(0.1, 0.3), 2.0, 10.0)
            tt += rng.uniform(20, 40)
        labels.append(dict(type="plm_series", category="negative", onset_s=t0,
                           offset_s=t0 + dur, amplitude=np.nan))
    return out, labels


def apply_device_artifacts(t, sc, acc, skin, ppg_amp, v, rng):
    """Wrist-device artifacts on EDA, skin temperature, PPG amplitude and movement."""
    T = t[-1] + (t[1] - t[0])
    s = v["artifact_rate_scale"]
    sc, acc, skin, ppg_amp = sc.copy(), acc.copy(), skin.copy(), ppg_amp.copy()
    labels = []
    dt = t[1] - t[0]
    for t0 in _poisson_times(rng, s * v["eda_motion_rate"], T):
        amp = v["eda_motion_amp"] * 2.0 ** rng.normal() * rng.choice([1, 1, -1])
        step = rng.random() < 0.3  # sometimes a baseline step that decays slowly
        _add(sc, t, t0, amp, 0.15, 30.0 if step else rng.uniform(1, 5))
        _add(acc, t, t0, rng.uniform(0.1, 0.8), 0.2, 1.5)
        _add(ppg_amp, t, t0, rng.uniform(-0.5, 0.5), 0.2, 3.0)
        labels.append(dict(type="eda_motion", category="artifact", onset_s=t0,
                           offset_s=t0 + (90 if step else 15), amplitude=amp))
    for t0 in _poisson_times(rng, s * v["eda_contact_loss_rate"], T):
        dur = rng.uniform(5, 180)
        i0, i1 = int(t0 / dt), min(int((t0 + dur) / dt), len(t))
        sc[i0:i1] = rng.uniform(0.0, 0.1) + rng.normal(0, 0.005, i1 - i0)
        ppg_amp[i0:i1] = rng.uniform(0.0, 0.05)
        # skin sensor cools toward ambient, then re-warms after re-contact
        tt = t[i0:] - t[i0]
        drop = np.where(tt < dur, 1 - np.exp(-tt / 60.0),
                        (1 - np.exp(-dur / 60.0)) * np.exp(-(tt - dur) / 120.0))
        skin[i0:] -= (skin[i0] - v["skin_ambient"]) * drop
        _add(sc, t, t0 + dur, rng.uniform(0.3, 1.5), 0.5, 20.0)
        _add(acc, t, t0, rng.uniform(0.1, 0.5), 0.2, 1.5)
        _add(acc, t, t0 + dur, rng.uniform(0.1, 0.5), 0.2, 1.5)
        labels.append(dict(type="device_contact_loss", category="artifact", onset_s=t0,
                           offset_s=t0 + dur + 120, amplitude=np.nan))
    sc = np.clip(sc, 0.0, v["eda_max"])
    q = v["eda_quantum"]
    sc = np.round(sc / q) * q
    acc = np.abs(acc + rng.normal(0, v["acc_noise_sd"], len(acc)))
    ppg_amp = np.maximum(ppg_amp, 0.0)
    return sc, acc, skin, ppg_amp, labels


def ectopic_beats(r_t, v, rng):
    """Physiological premature beats: R peak occurs early (compensatory pause follows).
    Returns new R times and a boolean mask of ectopic beats."""
    r = r_t.copy()
    ect = np.zeros(len(r), bool)
    for i in range(1, len(r) - 1):
        if rng.random() < v["ectopic_frac"]:
            r[i] = r[i - 1] + rng.uniform(0.55, 0.75) * (r[i] - r[i - 1])
            ect[i] = True
    return r, ect


def ppg_pulses(r_t, ect, pat_ms, t, acc, contact_loss, v, rng):
    """Map R peaks to detected wrist pulses (PRV source) with PPG artifacts.

    pat_ms: pulse arrival time per beat; acc: movement on the t grid;
    contact_loss: list of (t0, t1) intervals with no PPG.
    """
    s = v["artifact_rate_scale"]
    p = r_t + pat_ms / 1000.0 + rng.normal(0, v["ppg_jitter_sd"] / 1000.0, len(r_t))
    dt = t[1] - t[0]
    a_at = acc[np.clip((p / dt).astype(int), 0, len(acc) - 1)]
    out, labels = [], []
    for i, pt in enumerate(p):
        if ect[i]:
            labels.append(dict(type="ectopic", category="artifact", onset_s=pt,
                               offset_s=pt + 3, amplitude=np.nan))
            if rng.random() < 0.5:  # premature beat: weak stroke, pulse often missed
                continue
        if a_at[i] > 0.05 and rng.random() < s * v["ppg_motion_corrupt_prob"]:
            if rng.random() < 0.5:
                labels.append(dict(type="ppg_motion", category="artifact", onset_s=pt,
                                   offset_s=pt + 2, amplitude=np.nan))
                continue
            pt = pt + rng.normal(0, 0.1)
        u = rng.random()
        if u < s * v["ppg_missed_frac"]:
            labels.append(dict(type="ppg_missed", category="artifact", onset_s=pt,
                               offset_s=pt + 2, amplitude=np.nan))
            continue
        if u < s * (v["ppg_missed_frac"] + v["ppg_extra_frac"]) and out:
            out.append(out[-1] + rng.uniform(0.3, 0.7) * (pt - out[-1]))
            labels.append(dict(type="ppg_extra", category="artifact", onset_s=out[-1],
                               offset_s=out[-1] + 2, amplitude=np.nan))
        out.append(pt)
    out = np.sort(np.array(out))
    T = t[-1]
    gaps = list(contact_loss)
    for t0 in _poisson_times(rng, s * v["ppg_dropout_rate"], T):
        dur = rng.uniform(10, 60)
        gaps.append((t0, t0 + dur))
        labels.append(dict(type="ppg_dropout", category="artifact", onset_s=t0,
                           offset_s=t0 + dur, amplitude=np.nan))
    for a, b in gaps:
        out = out[(out < a) | (out > b)]
    return out, labels
