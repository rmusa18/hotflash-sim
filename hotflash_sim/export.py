"""Reproducible dataset export with dev/test splits and hidden labels.

Layout (all derived from one master seed; same seed -> byte-identical files):

  <out>/manifest.json                 master seed, versions, scenario list, sha256 of every file
  <out>/<split>/<scenario_id>/
        signals_2hz.csv               observable channels only (what a detector sees)
        pulses.csv                    detected wrist PPG pulse times + inter-beat intervals (PRV)
        meta.json                     group, seed, hours, artifact/negative level
        params.json                   full parameter records used (value/unit/status/source)
  <out>/hidden/<split>/<scenario_id>/
        labels.csv                    every event: hot flashes, negatives, artifacts (opaque IDs)
        flash_details.csv             per-flash draws (amplitudes, lead/lag, NKB at onset ...)
        latent_2hz.csv                core temp, thresholds, KNDy states, vagal/sympathetic, clean EDA
        ecg_true.csv                  true R peaks (cardiac ground truth, incl. ectopic flag)
  <out>/figures/dev/...               dev figures (labels overlaid)
  <out>/hidden/figures/test/...       test figures (contain ground truth)
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict

import numpy as np
import pandas as pd

from .params import get_params, values, GROUPS
from .simulate import simulate_night

from . import __version__ as VERSION
FLOAT = "%.6g"
OBSERVABLE = ["t_s", "eda_uS", "skin_temp_C", "pr_bpm", "prv_rmssd_60s_ms", "ppg_amp",
              "acc_g", "resp"]
LATENT = ["t_s", "clock_h", "core_C", "sweat_thresh_C", "shiver_thresh_C", "kndy_dyn_nM",
          "kndy_nkb_nM", "kndy_firing_spm", "sc_phys_uS", "sc_hf_component_uS",
          "skin_phys_C", "ppg_amp_phys", "vagal_tone", "symp_index", "hr_bpm_true",
          "resp_rate_bpm"]
SUBJECT_SD = {"sweat_thresh_offset": 0.03, "hr_base": 5.0, "rsa_amp_base": 6.0,
              "scl_baseline": 1.0}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def causal_hr_rmssd(beats, t_grid, win=60.0):
    """Sample-and-hold pulse rate from the last clean inter-beat interval and
    trailing-window RMSSD, both strictly causal (only pulses at or before each
    sample). Implausible intervals are removed with a causal median filter."""
    bt = beats.t_s.to_numpy(); rr = beats.ibi_ms.to_numpy()
    med = pd.Series(rr).rolling(11, min_periods=3).median().to_numpy()
    ok = (rr > 300) & (rr < 2000) & (np.abs(rr - med) <= 0.25 * med)
    bt, rr = bt[ok], rr[ok]
    idx = np.searchsorted(bt, t_grid, side="right") - 1
    hr = np.where(idx >= 0, 60000.0 / rr[np.clip(idx, 0, None)], np.nan)
    d2 = np.r_[0.0, np.diff(rr) ** 2]
    c = np.cumsum(d2)
    lo = np.searchsorted(bt, t_grid - win, side="left")
    cnt = idx - lo  # number of successive differences in window
    s = np.where(idx >= 0, c[np.clip(idx, 0, None)], 0) - np.where(lo >= 0, c[np.clip(lo, 0, None)], 0)
    rmssd = np.where(cnt >= 10, np.sqrt(np.maximum(s, 0) / np.maximum(cnt, 1)), np.nan)
    return hr, rmssd


def to_output_rate(run):
    """Decimate the 4 Hz simulation grid to the output rate (default 2 Hz,
    one sample per 500 ms) by block averaging, and split observable/latent."""
    v = values(run["params"])
    s = run["signals"]
    dt = s.t_s.iloc[1] - s.t_s.iloc[0]
    blk = max(int(round(1.0 / (v["output_rate"] * dt))), 1)
    m = len(s) // blk * blk
    cols = [c for c in s.columns if c != "t_s"]
    avg = s[cols].iloc[:m].to_numpy().reshape(-1, blk, len(cols)).mean(1)
    d = pd.DataFrame(avg, columns=cols)
    d.insert(0, "t_s", s.t_s.iloc[:m:blk].to_numpy() + (blk - 1) * dt)  # sample time = block end
    hr, rm = causal_hr_rmssd(run["beats"], d.t_s.to_numpy())
    obs = pd.DataFrame({"t_s": d.t_s, "eda_uS": d.sc_uS, "skin_temp_C": d.skin_temp_C,
                        "pr_bpm": hr, "prv_rmssd_60s_ms": rm, "ppg_amp": d.ppg_amp,
                        "acc_g": d.acc_g, "resp": d.resp})
    lat = d.rename(columns={"hr_bpm": "hr_bpm_true"})[LATENT]
    return obs, lat


def plan_scenarios(master_seed, n_per_group, groups=GROUPS, hours=8.0,
                   artifact_levels=(0.5, 1.0, 2.0), negative_level=1.0):
    """Scenario specs for dev and test. Splits use independent child seeds of
    the master SeedSequence, so they never share a random stream."""
    root = np.random.SeedSequence(master_seed)
    specs = []
    for split, child in zip(("dev", "test"), root.spawn(2)):
        for gi, (g, gchild) in enumerate(zip(groups, child.spawn(len(groups)))):
            for i, sc in enumerate(gchild.spawn(n_per_group)):
                seed = int(sc.generate_state(1)[0])
                rng = np.random.default_rng(sc.spawn(1)[0])
                base = values(get_params(g))
                ov = {k: float(base[k] + rng.normal(0, sd)) for k, sd in SUBJECT_SD.items()}
                ov["rsa_amp_base"] = max(ov["rsa_amp_base"], 5.0)
                ov["scl_baseline"] = max(ov["scl_baseline"], 0.8)
                ov["artifact_rate_scale"] = float(artifact_levels[i % len(artifact_levels)])
                ov["negative_rate_scale"] = float(negative_level)
                sid = "scn_" + hashlib.sha1(f"{master_seed}:{split}:{g}:{i}".encode()).hexdigest()[:8]
                specs.append(dict(scenario_id=sid, split=split, group=g, index=i,
                                  seed=seed, hours=hours, overrides=ov))
    return specs


def export_scenario(spec, out):
    run = simulate_night(spec["group"], spec["hours"], seed=spec["seed"], **spec["overrides"])
    obs, lat = to_output_rate(run)
    vis = os.path.join(out, spec["split"], spec["scenario_id"])
    hid = os.path.join(out, "hidden", spec["split"], spec["scenario_id"])
    os.makedirs(vis, exist_ok=True); os.makedirs(hid, exist_ok=True)
    obs.to_csv(os.path.join(vis, "signals_2hz.csv"), index=False, float_format=FLOAT, lineterminator="\n")
    run["beats"].to_csv(os.path.join(vis, "pulses.csv"), index=False, float_format=FLOAT, lineterminator="\n")
    meta = {k: spec[k] for k in ("scenario_id", "split", "group", "seed", "hours")}
    meta.update(artifact_rate_scale=spec["overrides"]["artifact_rate_scale"],
                negative_rate_scale=spec["overrides"]["negative_rate_scale"],
                output_rate_hz=values(run["params"])["output_rate"],
                observable_columns=OBSERVABLE, simulator_version=VERSION)
    with open(os.path.join(vis, "meta.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    with open(os.path.join(vis, "params.json"), "w", newline="\n") as f:
        json.dump({k: asdict(p) for k, p in run["params"].items()}, f, indent=2, sort_keys=True)
    run["labels"].to_csv(os.path.join(hid, "labels.csv"), index=False, float_format=FLOAT, lineterminator="\n")
    run["events"].to_csv(os.path.join(hid, "flash_details.csv"), index=False, float_format=FLOAT, lineterminator="\n")
    lat.to_csv(os.path.join(hid, "latent_2hz.csv"), index=False, float_format=FLOAT, lineterminator="\n")
    run["ecg_true"].to_csv(os.path.join(hid, "ecg_true.csv"), index=False, float_format=FLOAT, lineterminator="\n")
    return run, obs, lat


def write_manifest(out, master_seed, specs, settings):
    files = {}
    for root, _, names in os.walk(out):
        for nm in sorted(names):
            p = os.path.join(root, nm)
            rel = os.path.relpath(p, out)
            if rel == "manifest.json":
                continue
            files[rel] = _sha(p)
    pkg = os.path.dirname(__file__)
    code = {nm: _sha(os.path.join(pkg, nm)) for nm in sorted(os.listdir(pkg)) if nm.endswith(".py")}
    man = {"simulator_version": VERSION, "master_seed": master_seed, "settings": settings,
           "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "code_sha256": code,
           "scenarios": [{k: s[k] for k in ("scenario_id", "split", "group", "seed")} for s in specs],
           "files_sha256": dict(sorted(files.items()))}
    with open(os.path.join(out, "manifest.json"), "w", newline="\n") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    return man


def verify(out):
    """Re-hash every file and compare with the manifest."""
    with open(os.path.join(out, "manifest.json")) as f:
        man = json.load(f)
    bad = [rel for rel, h in man["files_sha256"].items()
           if not os.path.exists(os.path.join(out, rel)) or _sha(os.path.join(out, rel)) != h]
    return bad
