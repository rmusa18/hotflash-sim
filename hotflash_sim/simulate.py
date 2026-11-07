"""Coupled simulator.

Layers (upstream -> downstream):

1. KNDy pulse generator  - gamma renewal process; a fraction of pulses
   (Prague 2019 coupling) produce central sympathetic activation a(t).
2. Freedman threshold model - core temp Tc(t) (circadian + OU noise + post-flash
   heat loss) vs sweating threshold Tsw(t) = Tsw0 - k * a(t).  The flash hazard
   is a soft (sigmoid) threshold on Tc - Tsw, with a refractory period.
3. Effectors - per-flash skin conductance response, vagal withdrawal (correlated
   with SC amplitude, de Zambotti 2013), sympathetic HR rise, respiration rise.
4. Heart - beat-by-beat RR intervals: mean RR from HR(t), RSA scaled by vagal
   tone, 0.1 Hz LF component, white noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, lfilter

from .params import get_params, values
from .kndy import simulate_kndy
from .confounders import (generate_negatives, apply_device_artifacts,
                          ectopic_beats, ppg_pulses)
import hashlib

_KNDY_CACHE: dict = {}


def cached_kndy(v, minutes, kseed):
    """KNDy trace for a seed; cached because calibration reruns the same seeds."""
    keys = sorted(k for k in v if k.startswith("kndy_") and k != "kndy_threshold_drop")
    key = (tuple(v[k] for k in keys), minutes, None if kseed is None else tuple(kseed))
    if kseed is None or key not in _KNDY_CACHE:
        out = simulate_kndy(v, minutes, np.random.default_rng(kseed))
        if kseed is None:
            return out
        if len(_KNDY_CACHE) > 400:
            _KNDY_CACHE.clear()
        _KNDY_CACHE[key] = out
    return _KNDY_CACHE[key]


def _kernel(t, tau_r, tau_d):
    """Difference-of-exponentials kernel, peak normalised to 1, zero for t<0."""
    t = np.asarray(t, float)
    tp = tau_r * tau_d / (tau_d - tau_r) * np.log(tau_d / tau_r)
    peak = np.exp(-tp / tau_d) - np.exp(-tp / tau_r)
    k = (np.exp(-t / tau_d) - np.exp(-t / tau_r)) / peak
    return np.where(t >= 0, k, 0.0)


def _ou(n, dt, tau, sd, rng):
    """Exact-discretised Ornstein-Uhlenbeck path, stationary SD `sd`."""
    a = np.exp(-dt / tau)
    e = rng.normal(0, sd * np.sqrt(1 - a * a), n)
    e[0] = rng.normal(0, sd)
    return lfilter([1.0], [1.0, -a], e)


def _add_events(t, onsets, amps, tau_r, tau_d, span=8.0):
    """Sum of kernels at `onsets` scaled by `amps` (evaluated on a local span)."""
    out = np.zeros_like(t)
    dt = t[1] - t[0]
    w = int(span * tau_d / dt)
    for t0, a in zip(onsets, amps):
        i0 = max(int(np.floor(t0 / dt)), 0)
        i1 = min(i0 + w, len(t))
        if i0 >= len(t):
            continue
        out[i0:i1] += a * _kernel(t[i0:i1] - t0, tau_r, tau_d)
    return out


def simulate_night(group="menopausal_symptomatic", hours=8.0, start_clock=23.0,
                   seed=None, params=None, **overrides):
    """Simulate one night for one person.

    Returns a dict with:
      signals : DataFrame on the dt grid (core temp, thresholds, KNDy activation,
                SC, vagal tone, sympathetic index, HR, respiration)
      beats   : DataFrame of beat times and RR intervals (ms)
      events  : DataFrame of ground-truth flashes with per-event draws
      pulses  : KNDy pulse times (peaks of population firing)
      params  : the parameter records used
    """
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    rng = np.random.default_rng([seed, 0])  # physiology stream
    P = params if params is not None else get_params(group, **overrides)
    v = values(P)
    dt = v["dt"]
    T = hours * 3600.0
    t = np.arange(0.0, T, dt)
    n = len(t)

    # ---- 1. KNDy pulse generator (Voliotis et al. 2021 mean-field model) ----
    # separate RNG stream so the KNDy trace depends only on the seed (cacheable)
    kseed = None if seed is None else [seed, 7]
    tk, Dk, Nk, Fk, pk = cached_kndy(v, hours * 60, kseed)
    tmin = t / 60
    kndy_D = np.interp(tmin, tk, Dk)
    kndy_N = np.interp(tmin, tk, Nk)
    kndy_F = np.interp(tmin, tk, Fk)
    pulses = pk * 60
    act = kndy_N / v["kndy_N_ref"]  # normalised NKB drives threshold lowering

    # ---- 2. Thermoregulation + flash trigger -------------------------------
    clock = (start_clock + t / 3600.0) % 24
    circ = -v["core_circ_amp"] * np.cos(2 * np.pi * (clock - v["core_circ_nadir_h"]) / 24)
    core_free = v["core_mean"] + circ + _ou(n, dt, v["core_ou_tau"] * 60, v["core_ou_sd"], rng)
    sweat0 = (v["core_mean"] + v["sweat_thresh_offset"]
              + v["neutral_zone_width"] / 2 + v["estrogen_sweat_shift"])
    sweat = sweat0 - v["kndy_threshold_drop"] * act
    shiver = sweat - v["neutral_zone_width"]

    # Event-driven: hazard is evaluated vectorised between flashes; each flash
    # adds a heat-loss transient (core drop) to the core temperature.
    lam_max = v["trigger_max_rate"] / 60 * dt
    soft = v["trigger_softness"]
    ref_n = int(v["refractory"] * 60 / dt)
    tau_d = v["flash_core_drop_tau"] * 60
    u_draw = rng.random(n)
    core = core_free.copy()
    onsets_idx = []
    start = int(v["burn_in"] * 60 / dt)  # skip start-up transient
    while start < n:
        x = np.clip((core[start:] - sweat[start:]) / soft, -50, 50)
        h = lam_max / (1 + np.exp(-x))
        hit = np.flatnonzero(u_draw[start:] < h)
        if hit.size == 0:
            break
        i = start + hit[0]
        onsets_idx.append(i)
        core[i:] -= v["flash_core_drop"] * _kernel(t[i:] - t[i], 5 * 60.0, tau_d)
        start = i + ref_n
    onsets = t[onsets_idx] if onsets_idx else np.array([])
    k_ev = len(onsets)

    # ---- 3. Per-flash effector draws ---------------------------------------
    rho = v["vagal_sc_corr"]
    z = rng.normal(size=k_ev)
    sc_amp = v["hf_sc_amp_median"] * v["hf_sc_amp_gsd"] ** z
    eps = rng.normal(size=k_ev)
    # vagal tone *change* is correlated with log SC amplitude at r = rho
    dvagal = -v["vagal_withdrawal_mean"] + v["vagal_withdrawal_sd"] * (
        rho * z + np.sqrt(1 - rho**2) * eps)
    withdrawal = np.clip(-dvagal, 0.0, 0.9)
    dhr = rng.uniform(v["hf_dhr_low"], v["hf_dhr_high"], k_ev)
    f_v = v["hr_vagal_fraction"]
    scale = withdrawal / max(v["vagal_withdrawal_mean"], 1e-9)
    dhr_v = f_v * dhr * scale
    dhr_s = (1 - f_v) * dhr
    leads = rng.random(k_ev) < v["hr_pre_onset_frac"]
    hr_off = np.where(leads,
                      -np.abs(rng.normal(v["hr_lead_mean"], v["hr_lead_sd"], k_ev)),
                      np.abs(rng.normal(v["hr_lag_mean"], v["hr_lead_sd"] / 2, k_ev)))
    hr_on = onsets + hr_off

    win = int(600 / dt)  # 10 min look-back for core-rise check
    core_rise = np.array([core_free[i] - core_free[max(i - win, 0)] for i in onsets_idx])
    events = pd.DataFrame({
        "onset_s": onsets, "clock_h": (start_clock + onsets / 3600) % 24,
        "sc_amp_uS": sc_amp, "vagal_withdrawal": withdrawal, "dHR_bpm": dhr,
        "dHR_vagal": dhr_v, "dHR_symp": dhr_s, "hr_onset_s": hr_on,
        "hr_leads": leads, "core_rise_10min": core_rise,
        "core_rise_before": core_rise > 0.0,
        "nkb_at_onset_nM": kndy_N[onsets_idx] if k_ev else np.array([]),
        "min_since_pulse": np.array([(o - pulses[pulses <= o].max()) / 60
                                     if (pulses <= o).any() else np.nan for o in onsets]),
    })

    # ---- cardiac / vagal / respiratory continuous states -------------------
    cr, cd = v["cardiac_tau_rise"], v["cardiac_tau_decay"]
    vagal = 1.0 - _add_events(t, hr_on, withdrawal, cr, cd)
    symp = _add_events(t, hr_on, dhr_s / v["hf_dhr_high"], cr, cd)
    hr = (v["hr_base"] + _ou(n, dt, 20 * 60, 2.0, rng)
          + _add_events(t, hr_on, dhr_s, cr, cd) + _add_events(t, hr_on, dhr_v, cr, cd))
    rr_rate = (v["resp_rate"] + _ou(n, dt, 10 * 60, v["resp_rate_sd"], rng)
               + _add_events(t, hr_on, np.full(k_ev, v["hf_resp_increase"]), cr, cd))
    # ---- negative examples (physiological mimics), own RNG stream ---------
    neg, neg_labels = generate_negatives(t, v, np.random.default_rng([seed, 1]))
    hr = hr + neg["dhr"]
    vagal = np.clip(vagal + neg["dvagal"], 0.05, 1.2)
    rr_rate = rr_rate + neg["dresp"]
    phase = 2 * np.pi * np.cumsum(rr_rate / 60) * dt
    resp = np.sin(phase)

    # ---- skin conductance --------------------------------------------------
    scl = v["scl_baseline"] + _ou(n, dt, v["scl_drift_tau"] * 60, v["scl_drift_sd"], rng)
    scl = np.maximum(scl, 0.5)
    ns_n = rng.poisson(v["nsscr_rate"] / 60 * T)
    imp = np.zeros(n)
    idx = rng.integers(0, n, ns_n)
    np.add.at(imp, idx, v["nsscr_amp_median"] * v["nsscr_amp_gsd"] ** rng.normal(size=ns_n))
    k_ns = _kernel(np.arange(0, 40, dt), 1.0, 4.0)
    nsscr = fftconvolve(imp, k_ns)[:n]
    sc_hf = _add_events(t, onsets, sc_amp, v["hf_sc_tau_rise"], v["hf_sc_tau_decay"])
    sc_phys = scl + nsscr + sc_hf + neg["dsc"] + rng.normal(0, v["sc_noise_sd"], n)

    # ---- wrist skin temperature and PPG amplitude (own RNG stream) ---------
    rng_w = np.random.default_rng([seed, 3])
    k_ev = len(onsets)
    skin_rise = v["skin_flash_rise_median"] * 1.5 ** rng_w.normal(size=k_ev)
    skin_phys = (v["skin_base"] + _ou(n, dt, v["skin_ou_tau"] * 60, v["skin_ou_sd"], rng_w)
                 + _add_events(t, onsets + 30, skin_rise, v["skin_flash_tau_rise"],
                               v["skin_flash_tau_decay"]) + neg["dskin"])
    vasodil = _add_events(t, onsets, v["ppg_amp_flash_median"] * 1.5 ** rng_w.normal(size=k_ev),
                          60.0, 400.0) + neg["dvasodil"]
    ppg_amp_phys = (np.exp(_ou(n, dt, 20 * 60, v["ppg_amp_ou_sd"], rng_w)) * (1 + vasodil)
                    * np.clip(1 - neg["dvasocon"], 0.1, 1))

    # ---- measurement artifacts (own RNG stream) -----------------------------
    rng_art = np.random.default_rng([seed, 2])
    sc, acc, skin, ppg_amp, dev_labels = apply_device_artifacts(
        t, sc_phys, neg["acc"], skin_phys, ppg_amp_phys, v, rng_art)
    contact = [(d["onset_s"], d["offset_s"] - 120) for d in dev_labels
               if d["type"] == "device_contact_loss"]

    # ---- cardiac beats (true R peaks) ---------------------------------------
    lf_phase = rng.uniform(0, 2 * np.pi)
    beats, rrs = [], []
    tb = rng.uniform(0, 1)
    while tb < T:
        i = min(int(tb / dt), n - 1)
        rr = (60000.0 / hr[i]
              + v["rsa_amp_base"] * vagal[i] * np.sin(phase[i])
              + v["lf_amp"] * np.sin(2 * np.pi * 0.1 * tb + lf_phase)
              + rng.normal(0, v["rr_noise_sd"]))
        rr = max(rr, 300.0)
        beats.append(tb)
        rrs.append(rr)
        tb += rr / 1000.0
    r_t, ect = ectopic_beats(np.array(beats), v, rng_art)
    ecg_true = pd.DataFrame({"t_s": r_t[1:], "rr_ms": np.diff(r_t) * 1000.0,
                             "ectopic": ect[1:]})

    # ---- PPG pulses: R peak + pulse arrival time (PEP + PTT) ---------------
    ri = np.clip((r_t / dt).astype(int), 0, n - 1)
    pat = (v["pat_base"] - v["pep_drop_symp"] * symp[ri]
           + v["ptt_resp_mod"] * np.sin(phase[ri] + 1.0))
    ppg_t, ppg_labels = ppg_pulses(r_t, ect, pat, t, acc, contact, v, rng_art)
    beats_df = pd.DataFrame({"t_s": ppg_t[1:], "ibi_ms": np.diff(ppg_t) * 1000.0})
    eda_labels, rr_labels = dev_labels, ppg_labels

    # ---- ground-truth label table with opaque event IDs --------------------
    hf_labels = [dict(type="hot_flash", category="hot_flash", onset_s=o,
                      offset_s=o + 600, amplitude=a)
                 for o, a in zip(onsets, sc_amp)]
    labels = pd.DataFrame(hf_labels + neg_labels + eda_labels + rr_labels)
    if len(labels):
        labels = labels.sort_values("onset_s", kind="stable").reset_index(drop=True)
        labels.insert(0, "event_id", [
            "evt_" + hashlib.sha1(f"{seed}:{k}:{r.type}:{r.onset_s:.3f}".encode()).hexdigest()[:10]
            for k, r in enumerate(labels.itertuples())])
    else:
        labels = pd.DataFrame(columns=["event_id", "type", "category", "onset_s",
                                       "offset_s", "amplitude"])
    hf_ids = labels[labels.type == "hot_flash"].sort_values("onset_s").event_id.to_numpy()
    events.insert(0, "event_id", hf_ids)

    signals = pd.DataFrame({
        "t_s": t, "clock_h": clock, "core_C": core, "sweat_thresh_C": sweat,
        "shiver_thresh_C": shiver, "kndy_dyn_nM": kndy_D, "kndy_nkb_nM": kndy_N,
        "kndy_firing_spm": kndy_F, "sc_uS": sc, "sc_phys_uS": sc_phys, "acc_g": acc,
        "skin_temp_C": skin, "skin_phys_C": skin_phys, "ppg_amp": ppg_amp,
        "ppg_amp_phys": ppg_amp_phys,
        "sc_hf_component_uS": sc_hf, "vagal_tone": vagal, "symp_index": symp,
        "hr_bpm": hr, "resp_rate_bpm": rr_rate, "resp": resp,
    })
    # "beats" = detected wrist PPG pulses (PRV source); "ecg_true" = hidden R peaks
    return {"signals": signals, "beats": beats_df, "ecg_true": ecg_true,
            "events": events, "labels": labels, "seed": seed,
            "pulses": pd.DataFrame({"t_s": pulses}),
            "params": P, "group": group, "hours": hours}


def simulate_cohort(group, n_subjects=10, hours=8.0, seed=0,
                    between_subject_sd=None, **overrides):
    """Simulate several people; key parameters vary between subjects.

    between_subject_sd maps parameter name -> SD of a normal perturbation
    (absolute units). Defaults are ASSUMED.
    """
    rng = np.random.default_rng(seed)
    bsd = between_subject_sd or {"sweat_thresh_offset": 0.03, "hr_base": 5.0,
                                 "rsa_amp_base": 6.0, "scl_baseline": 1.0}
    runs = []
    for s in range(n_subjects):
        ov = dict(overrides)
        base = values(get_params(group, **overrides))
        for k, sd in bsd.items():
            ov[k] = base[k] + rng.normal(0, sd)
        ov["rsa_amp_base"] = max(ov.get("rsa_amp_base", base["rsa_amp_base"]), 5.0)
        ov["scl_baseline"] = max(ov.get("scl_baseline", base["scl_baseline"]), 0.8)
        r = simulate_night(group, hours=hours, seed=int(rng.integers(1 << 31)), **ov)
        r["subject"] = s
        runs.append(r)
    return runs


def calibrate(target=None, group="menopausal_symptomatic", n_nights=12,
              hours=8.0, lo=-0.1, hi=0.4, iters=14, seed=123):
    """Bisect `sweat_thresh_offset` so the mean flash count per night matches
    `target` (default: de Zambotti 2014, 222/63 per night). Uses common random
    numbers so the count is monotone in the offset."""
    base = get_params(group)
    target = target if target is not None else base["target_rate_night"].value

    def mean_count(off):
        return np.mean([len(simulate_night(group, hours, seed=seed + k,
                                            sweat_thresh_offset=off)["events"])
                        for k in range(n_nights)])
    for _ in range(iters):
        mid = (lo + hi) / 2
        if mean_count(mid) > target:
            lo = mid
        else:
            hi = mid
    off = (lo + hi) / 2
    return off, mean_count(off)


def _night_stats(group, hours, seeds, **ov):
    evs = [simulate_night(group, hours, seed=s, **ov)["events"] for s in seeds]
    ev = pd.concat(evs, ignore_index=True)
    frac = ev.core_rise_before.mean() if len(ev) else np.nan
    return len(ev) / len(seeds), frac


def calibrate_joint(group="menopausal_symptomatic", drops=(0.0, 0.1, 0.2, 0.3, 0.45, 0.6),
                    n_nights=40, hours=8.0, seed=123, verbose=True):
    """Fit kndy_threshold_drop and sweat_thresh_offset together so that
    flashes/night = target_rate_night and the fraction preceded by a core-temp
    rise = target_frac_core_rise. For each drop, the offset is bisected to hit
    the rate; the drop is then interpolated to hit the fraction."""
    P = get_params(group)
    rate_t = P["target_rate_night"].value
    frac_t = P["target_frac_core_rise"].value
    seeds = [seed + k for k in range(n_nights)]
    rows = []
    for dr in drops:
        lo, hi = -0.2, 2.5
        for _ in range(13):
            mid = (lo + hi) / 2
            r, _ = _night_stats(group, hours, seeds, sweat_thresh_offset=mid,
                                kndy_threshold_drop=dr)
            lo, hi = (mid, hi) if r > rate_t else (lo, mid)
        off = (lo + hi) / 2
        r, f = _night_stats(group, hours, seeds, sweat_thresh_offset=off,
                            kndy_threshold_drop=dr)
        rows.append((dr, off, r, f))
        if verbose:
            print(f"drop={dr:.2f}  offset={off:.4f}  rate={r:.2f}  frac_core_rise={f:.3f}")
    tab = pd.DataFrame(rows, columns=["drop", "offset", "rate", "frac"])
    # interpolate the crossing of frac = target (frac decreases with drop)
    f = tab.frac.to_numpy()
    k = np.flatnonzero((f[:-1] - frac_t) * (f[1:] - frac_t) <= 0)
    if len(k) == 0:
        j = int(np.argmin(np.abs(f - frac_t)))
        return tab["drop"][j], tab["offset"][j], tab
    j = k[0]
    w = (frac_t - f[j]) / (f[j + 1] - f[j])
    drop = tab["drop"][j] + w * (tab["drop"][j + 1] - tab["drop"][j])
    off = tab["offset"][j] + w * (tab["offset"][j + 1] - tab["offset"][j])
    return drop, off, tab
