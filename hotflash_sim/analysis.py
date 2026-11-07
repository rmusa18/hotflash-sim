"""Detection, HRV metrics, event-locked statistics and literature checks."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- detection --
def detect_freedman(signals, rise_uS=2.0, window_s=30.0, lockout_s=300.0):
    """Freedman sternal-SC criterion: rise >= rise_uS within window_s.
    Returns detected onset times (s): start of the qualifying rise."""
    t = signals["t_s"].to_numpy()
    sc = signals["sc_uS"].to_numpy()
    dt = t[1] - t[0]
    w = int(window_s / dt)
    # rise relative to the running minimum over the preceding window
    from scipy.ndimage import minimum_filter1d
    runmin = minimum_filter1d(sc, size=w + 1, origin=w // 2)
    rise = sc - runmin
    hits = np.flatnonzero(rise >= rise_uS)
    out, last = [], -np.inf
    for i in hits:
        if t[i] - last > lockout_s:
            # onset = location of the minimum in the preceding window
            j0 = max(i - w, 0)
            out.append(t[j0 + np.argmin(sc[j0:i + 1])])
            last = t[i]
    return np.array(out)


def match_events(true_onsets, det_onsets, tol_s=120.0):
    """Greedy one-to-one matching; returns TP, FP, FN and per-match latency."""
    used = set()
    lat = []
    for tt in true_onsets:
        best = None
        for k, d in enumerate(det_onsets):
            if k not in used and abs(d - tt) <= tol_s:
                if best is None or abs(d - tt) < abs(det_onsets[best] - tt):
                    best = k
        if best is not None:
            used.add(best)
            lat.append(det_onsets[best] - tt)
    tp = len(lat)
    return {"TP": tp, "FN": len(true_onsets) - tp, "FP": len(det_onsets) - tp,
            "latency_s": np.array(lat)}


# ------------------------------------------------- PRV (pulse-rate variability) --
def ivcol(df):
    """Interval column: ibi_ms for PPG pulses, rr_ms for ECG beats."""
    return "ibi_ms" if "ibi_ms" in df else "rr_ms"


def clean_intervals(b, lo=300.0, hi=2000.0, rel=0.25):
    """Standard PRV preprocessing: drop physiologically implausible intervals and
    those deviating > rel from the local median (removes missed/extra pulses)."""
    c = ivcol(b)
    x = b[c].to_numpy()
    med = pd.Series(x).rolling(11, center=True, min_periods=3).median().to_numpy()
    ok = (x > lo) & (x < hi) & (np.abs(x - med) <= rel * med)
    return b[ok]

def hrv_window(beats, t0, t1):
    """PRV in [t0, t1): RMSSD (ms), mean pulse rate (bpm), HF power (0.15-0.4 Hz, ms^2).
    Intervals are cleaned first (see clean_intervals)."""
    b = beats[(beats.t_s >= t0) & (beats.t_s < t1)]
    b = clean_intervals(b) if len(b) else b
    if len(b) < 10:
        return np.nan, np.nan, np.nan
    rr = b[ivcol(b)].to_numpy()
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    hr = 60000 / rr.mean()
    # standard approach: cubic-spline resample to 4 Hz, Hann window, periodogram
    from scipy.interpolate import CubicSpline
    from scipy.signal import periodogram
    tb = b.t_s.to_numpy()
    tu = np.arange(tb[0], tb[-1], 0.25)
    x = CubicSpline(tb, rr)(tu)
    f, psd = periodogram(x, fs=4.0, window="hann", detrend="linear")
    band = (f >= 0.15) & (f < 0.40)
    hf = psd[band].sum() * (f[1] - f[0])
    return rmssd, hr, hf


def hrv_series(beats, win_s=60.0, step_s=30.0, t_end=None):
    t_end = t_end or beats.t_s.iloc[-1]
    rows = []
    for t0 in np.arange(0, t_end - win_s, step_s):
        r, h, f = hrv_window(beats, t0, t0 + win_s)
        rows.append((t0 + win_s / 2, r, h, f))
    return pd.DataFrame(rows, columns=["t_s", "rmssd_ms", "hr_bpm", "hf_ms2"])


# ------------------------------------------------------- event-locked analysis --
def event_windows(run, pre=(-180, -60), during=(0, 120), post=(300, 420)):
    """Per-flash HRV in pre / during / post windows (seconds rel. SC onset).
    Mirrors the windowed designs of de Zambotti 2013 and Baker 2019."""
    rows = []
    for k, ev in run["events"].iterrows():
        for name, (a, b) in {"pre": pre, "during": during, "post": post}.items():
            r, h, f = hrv_window(run["beats"], ev.onset_s + a, ev.onset_s + b)
            rows.append({"subject": run.get("subject", 0), "event": k, "window": name,
                         "rmssd_ms": r, "hr_bpm": h, "hf_ms2": f,
                         "sc_amp_uS": ev.sc_amp_uS})
    return pd.DataFrame(rows)


def event_locked_average(runs, column, span=(-300, 600), baseline=True):
    """Mean +/- SD of a continuous signal around SC onsets, across all events."""
    traces = []
    for r in runs:
        s = r["signals"]
        dt = s.t_s.iloc[1] - s.t_s.iloc[0]
        y = s[column].to_numpy()
        for t0 in r["events"].onset_s:
            i0, i1 = int((t0 + span[0]) / dt), int((t0 + span[1]) / dt)
            if i0 >= 0 and i1 <= len(y):
                seg = y[i0:i1]
                if baseline:  # subtract mean of the pre-onset part of the span
                    seg = seg - y[i0:int(t0 / dt)].mean()
                traces.append(seg)
    if not traces:
        return None
    tr = np.vstack(traces)
    lag = np.arange(tr.shape[1]) * dt + span[0]
    return lag, tr.mean(0), tr.std(0), len(tr)


def mixed_effects(ew, outcome="hr_bpm"):
    """Linear mixed model outcome ~ window with random intercept per subject."""
    import statsmodels.formula.api as smf
    d = ew.dropna(subset=[outcome]).copy()
    d["window"] = pd.Categorical(d.window, ["pre", "during", "post"])
    if d.subject.nunique() < 2:
        return None
    m = smf.mixedlm(f"{outcome} ~ C(window)", d, groups=d["subject"]).fit(reml=True)
    return m


# ------------------------------------------------------------ validation --
def literature_checks(runs, detect_runs=None):
    """Compare simulated statistics with literature targets.
    runs: clean recordings (no artifacts/mimics) for physiological checks.
    detect_runs: realistic recordings (artifacts + mimics) for detector checks."""
    P = runs[0]["params"]
    ev = pd.concat([r["events"] for r in runs], ignore_index=True)
    nights, hours = len(runs), runs[0]["hours"]
    symptomatic = runs[0]["group"] == "menopausal_symptomatic"
    out = []

    def add(name, sim, target, src):
        out.append({"check": name, "simulated": sim, "target": target, "source": src})

    add("flashes per night", len(ev) / nights * 8 / hours,
        P["target_rate_night"].value if symptomatic else "~0 (no hot flashes)",
        P["target_rate_night"].source if symptomatic else "group definition")
    iv = np.concatenate([np.diff(r["pulses"].t_s) / 60 for r in runs])
    add("KNDy pulse interval (min, mean; CV)", f"{iv.mean():.1f}; {iv.std() / iv.mean():.2f}",
        P["target_pulse_interval"].value, "assumed human LH pulse interval")
    if len(ev) and symptomatic:
        add("frac. preceded by core-temp rise", ev.core_rise_before.mean(),
            P["target_frac_core_rise"].value, P["target_frac_core_rise"].source)
        # pulse locking: flashes within 15 min after a KNDy pulse vs chance
        win = 15.0
        locked = np.mean(ev.min_since_pulse < win)
        chance = np.mean([min(win, 60) / np.mean(np.diff(r["pulses"].t_s) / 60) for r in runs])
        add("flashes <15 min after KNDy pulse onset (chance)", f"{locked:.2f} ({min(chance, 1):.2f})",
            "weak association, mean P=0.24", P["target_pulse_locking"].source)
        add("frac. HR rise before SC onset", ev.hr_leads.mean(),
            P["hr_pre_onset_frac"].value, P["hr_pre_onset_frac"].source)
        add("HR rise range (bpm, 5-95th pct)",
            f"{np.percentile(ev.dHR_bpm, 5):.1f}-{np.percentile(ev.dHR_bpm, 95):.1f}",
            f"{P['hf_dhr_low'].value:g}-{P['hf_dhr_high'].value:g}", P["hf_dhr_low"].source)
        r, n = measured_vagal_sc_r([dict(r, subject=k) for k, r in enumerate(runs)])
        add(f"r(PRV dlog HF, log SC amp), n={n}", r, -0.56,
            "de Zambotti et al. 2013 (PMC3713094)")
    for label, rs in (("clean", runs), ("realistic", detect_runs)):
        if not rs:
            continue
        det = [match_events(r["events"].onset_s.to_numpy(), detect_freedman(r["signals"]))
               for r in rs]
        tp = sum(d["TP"] for d in det); fn = sum(d["FN"] for d in det)
        fp = sum(d["FP"] for d in det)
        if symptomatic:
            add(f"Freedman detector sensitivity ({label})",
                tp / (tp + fn) if tp + fn else "n/a", "~0.9 (93% acc.)",
                "Forouzanfar 2018 (rule-based, vs experts)")
            add(f"Freedman detector PPV ({label})", tp / (tp + fp) if tp + fp else "n/a", "-", "")
        add(f"detector false positives/night ({label})", fp / len(rs), "-", "")
    return pd.DataFrame(out)


def measured_vagal_sc_r(runs):
    """Correlation between per-flash change in log HF-PRV (during vs pre) and
    log SC amplitude, measured from the detected pulses (as in de Zambotti 2013)."""
    ew = pd.concat([event_windows(r) for r in runs if len(r["events"])], ignore_index=True)
    if not len(ew):
        return np.nan, 0
    piv = ew.pivot_table(index=["subject", "event"], columns="window",
                         values=["hf_ms2", "sc_amp_uS"])
    d = np.log(piv[("hf_ms2", "during")]) - np.log(piv[("hf_ms2", "pre")])
    a = np.log(piv[("sc_amp_uS", "pre")])
    ok = np.isfinite(d) & np.isfinite(a)
    return (np.corrcoef(d[ok], a[ok])[0, 1] if ok.sum() > 5 else np.nan), int(ok.sum())


def calibrate_vagal_corr(target=-0.56, n_nights=60, seed=500, iters=7, verbose=True):
    """Bisect the generative vagal_sc_corr so the PRV-measured r equals target.
    Clean recordings (no artifacts/negatives), symptomatic group."""
    from .simulate import simulate_night
    lo, hi = -0.99, target  # measured |r| <= generative |r|
    def meas(rho):
        runs = []
        for k in range(n_nights):
            r = simulate_night("menopausal_symptomatic", seed=seed + k, vagal_sc_corr=rho,
                               artifact_rate_scale=0, negative_rate_scale=0)
            r["subject"] = k
            runs.append(r)
        return measured_vagal_sc_r(runs)
    for _ in range(iters):
        mid = (lo + hi) / 2
        r, n = meas(mid)
        if verbose:
            print(f"input={mid:.3f} measured={r:.3f} (n={n})", flush=True)
        lo, hi = (lo, mid) if r > target else (mid, hi)
    return (lo + hi) / 2
