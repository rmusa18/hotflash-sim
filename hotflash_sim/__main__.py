"""hotflash_sim command line.

One line, everything (dataset + figures + validation + manifest check):
    python -m hotflash_sim

Subcommands:
    run        build dev/test dataset, figures, literature checks   (default)
    stream     real-time 2 Hz stream of one scenario (JSON lines)
    score      score detector predictions against hidden labels
    verify     re-hash an exported dataset against its manifest
    params     print a group's parameter table
    calibrate  re-fit flash rate + core-rise (joint) and the vagal-SC correlation
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from .params import GROUPS, get_params, param_table, values


def cmd_run(a):
    from . import export, plots
    from .analysis import (event_windows, literature_checks, mixed_effects,
                           detect_freedman)
    from .simulate import simulate_night
    out = a.out
    ds = os.path.join(out, "dataset")
    os.makedirs(ds, exist_ok=True)
    ov = dict(kv.split("=") for kv in a.set)
    ov = {k: float(v) for k, v in ov.items()}

    # 1. dev / test scenarios from independent seed streams
    specs = export.plan_scenarios(a.seed, a.per_group, a.groups, a.hours,
                                  artifact_levels=tuple(a.artifact_levels))
    for sp in specs:
        sp["overrides"].update(ov)
    plotted = set()
    print(f"[1/3] exporting {len(specs)} scenarios (master seed {a.seed}) -> {ds}")
    for sp in specs:
        run, obs, lat = export.export_scenario(sp, ds)
        key = (sp["split"], sp["group"])
        if a.plot_all or key not in plotted:
            plotted.add(key)
            base = (os.path.join(ds, "figures", "dev") if sp["split"] == "dev"
                    else os.path.join(ds, "hidden", "figures", "test"))
            title = (f"{plots.GROUP_LABEL[sp['group']]} · {sp['scenario_id']} · "
                     f"artifacts x{sp['overrides']['artifact_rate_scale']:g}")
            plots.scenario_figures(obs, lat, run["beats"], run["labels"],
                                   os.path.join(base, sp["scenario_id"]), title,
                                   plots.GROUP_COLOR[sp["group"]])
        n_hf = int((run["labels"].category == "hot_flash").sum())
        print(f"   {sp['split']:4s} {sp['scenario_id']} {sp['group']:28s} "
              f"seed={sp['seed']:<10d} flashes={n_hf} labels={len(run['labels'])}")

    # 2. validation cohort: clean physiology + realistic detection
    print(f"[2/3] validation cohort: {a.validation_nights} nights per group")
    vdir = os.path.join(out, "validation")
    os.makedirs(vdir, exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence(a.seed).spawn(3)[2])
    checks, runs_by, ew_by = [], {}, {}
    for g in a.groups:
        seeds = rng.integers(0, 2**31, a.validation_nights)
        clean = [dict(simulate_night(g, a.hours, seed=int(s), artifact_rate_scale=0,
                                     negative_rate_scale=0, **ov), subject=k)
                 for k, s in enumerate(seeds)]
        real = [dict(simulate_night(g, a.hours, seed=int(s), **ov), subject=k)
                for k, s in enumerate(seeds)]
        runs_by[g] = clean
        c = literature_checks(clean, real); c.insert(0, "group", g); checks.append(c)
        if any(len(r["events"]) for r in clean):
            ew = pd.concat([event_windows(r) for r in clean], ignore_index=True)
            ew_by[g] = ew
            if ew.groupby(["subject", "event"]).ngroups >= 5:
                plots.event_locked(clean, os.path.join(vdir, f"event_locked_{g}.png"))
                for oc in ("hr_bpm", "rmssd_ms"):
                    m = mixed_effects(ew, oc)
                    if m is not None:
                        with open(os.path.join(vdir, f"mixedlm_{g}_{oc}.txt"), "w") as f:
                            f.write(str(m.summary()))
        else:
            ew_by[g] = pd.DataFrame()
        with open(os.path.join(vdir, f"params_{g}.md"), "w") as f:
            f.write(f"# Parameters: {g}\n\n" + param_table(g) + "\n")
    plots.hrv_windows(ew_by, os.path.join(vdir, "prv_windows.png"))
    plots.group_summary(runs_by, os.path.join(vdir, "group_summary.png"))
    ch = pd.concat(checks, ignore_index=True)
    ch.to_csv(os.path.join(vdir, "literature_checks.csv"), index=False)

    # 3. manifest + verification
    settings = dict(per_group=a.per_group, hours=a.hours, groups=a.groups,
                    artifact_levels=a.artifact_levels, overrides=ov)
    export.write_manifest(ds, a.seed, specs, settings)
    bad = export.verify(ds)
    print(f"[3/3] manifest written; verification: {'OK' if not bad else bad}")
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        print(ch[["group", "check", "simulated", "target"]].to_string(index=False))


def cmd_stream(a):
    from .stream import RealtimeStream, play
    ov = {k: float(v) for k, v in (kv.split("=") for kv in a.set)}
    s = RealtimeStream(a.group, a.seed, a.hours, **ov)
    print(f"# streaming {a.group} seed={a.seed} every {s.period * 1000:.0f} ms "
          f"(speed x{a.speed:g}); Ctrl-C to stop", file=sys.stderr)
    play(s, a.speed, a.out, a.detect, a.plot, a.plot_file, a.max_seconds, a.quiet)


def cmd_score(a):
    """Predictions CSV: scenario_id,t_s (predicted hot flash onsets)."""
    from .analysis import match_events, detect_freedman
    ds = a.data
    with open(os.path.join(ds, "manifest.json")) as f:
        man = json.load(f)
    scen = [s for s in man["scenarios"] if s["split"] == a.split]
    if a.pred:
        pred = pd.read_csv(a.pred)
    else:  # baseline: Freedman EDA criterion on the observable signals
        rows = []
        for s in scen:
            obs = pd.read_csv(os.path.join(ds, a.split, s["scenario_id"], "signals_2hz.csv"))
            for t0 in detect_freedman(obs.rename(columns={"eda_uS": "sc_uS"})):
                rows.append((s["scenario_id"], t0))
        pred = pd.DataFrame(rows, columns=["scenario_id", "t_s"])
        print("(no --pred given: scoring the Freedman 2 uS/30 s baseline)")
    tot = dict(TP=0, FP=0, FN=0); lat = []; fp_cause = {}
    for s in scen:
        lab = pd.read_csv(os.path.join(ds, "hidden", a.split, s["scenario_id"], "labels.csv"))
        truth = lab[lab.category == "hot_flash"].onset_s.to_numpy()
        p = np.sort(pred[pred.scenario_id == s["scenario_id"]].t_s.to_numpy())
        m = match_events(truth, p, a.tol)
        for k in ("TP", "FP", "FN"):
            tot[k] += m[k]
        lat += list(m["latency_s"])
        # attribute each unmatched prediction to the overlapping confounder
        for t0 in p:
            if np.any(np.abs(truth - t0) <= a.tol):
                continue
            ov = lab[(lab.category != "hot_flash") & (lab.onset_s - 30 <= t0) & (lab.offset_s + 30 >= t0)]
            prio = ["eda_motion", "device_contact_loss", "arousal", "ambient_heat",
                    "sc_storm", "plm_series"]  # causes that can move EDA
            hit = [p for p in prio if p in set(ov.type)]
            cause = hit[0] if hit else "unexplained (NS-SCR / noise)"
            fp_cause[cause] = fp_cause.get(cause, 0) + 1
    sens = tot["TP"] / max(tot["TP"] + tot["FN"], 1)
    ppv = tot["TP"] / max(tot["TP"] + tot["FP"], 1)
    print(f"split={a.split} scenarios={len(scen)} TP={tot['TP']} FP={tot['FP']} FN={tot['FN']}")
    print(f"sensitivity={sens:.3f} PPV={ppv:.3f} F1={2 * sens * ppv / max(sens + ppv, 1e-9):.3f} "
          f"median latency={np.median(lat) if lat else float('nan'):.1f} s")
    print("false positives by cause:", dict(sorted(fp_cause.items(), key=lambda x: -x[1])))


def cmd_verify(a):
    from .export import verify
    bad = verify(a.data)
    print("OK: all files match manifest" if not bad else f"MISMATCH: {bad}")
    sys.exit(1 if bad else 0)


def cmd_params(a):
    print(param_table(a.group))


def cmd_calibrate(a):
    from .simulate import calibrate_joint
    from .analysis import calibrate_vagal_corr
    d, o, tab = calibrate_joint()
    print(tab.to_string(index=False))
    print(f"kndy_threshold_drop={d:.4f} sweat_thresh_offset={o:.4f}")
    print(f"vagal_sc_corr={calibrate_vagal_corr():.4f}")
    print("Update the defaults in params.py (or pass via --set).")


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(prog="hotflash-sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="dataset + figures + validation (default)")
    r.add_argument("--seed", type=int, default=42, help="master seed")
    r.add_argument("--per-group", type=int, default=3, help="scenarios per group per split")
    r.add_argument("--hours", type=float, default=8.0)
    r.add_argument("--groups", nargs="*", default=GROUPS)
    r.add_argument("--artifact-levels", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    r.add_argument("--validation-nights", type=int, default=20)
    r.add_argument("--plot-all", action="store_true", help="figures for every scenario")
    r.add_argument("--out", default="outputs")
    r.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    r.add_argument("--quick", action="store_true",
                   help="fast preview: 1 scenario/group/split, 2 h nights, 3 validation nights")

    s = sub.add_parser("stream", help="real-time stream, one frame per 500 ms")
    s.add_argument("--group", default="menopausal_symptomatic", choices=GROUPS)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--hours", type=float, default=8.0)
    s.add_argument("--speed", type=float, default=1.0, help="1 = real time; 60 = 1 min per s")
    s.add_argument("--detect", action="store_true", help="online Freedman detector alerts")
    s.add_argument("--plot", action="store_true", help="live plot window")
    s.add_argument("--plot-file", help="live plot written to this PNG (headless)")
    s.add_argument("--out", help="write JSON lines here instead of stdout")
    s.add_argument("--max-seconds", type=float, help="stop after this much simulated time")
    s.add_argument("--quiet", action="store_true", help="no JSON output (plot/detect only)")
    s.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")

    c = sub.add_parser("score", help="score predictions against hidden labels")
    c.add_argument("--data", default="outputs/dataset")
    c.add_argument("--split", default="dev", choices=["dev", "test"])
    c.add_argument("--pred", help="CSV with scenario_id,t_s; omit to score the baseline")
    c.add_argument("--tol", type=float, default=120.0, help="match tolerance (s)")

    v = sub.add_parser("verify", help="check dataset files against manifest")
    v.add_argument("--data", default="outputs/dataset")

    p = sub.add_parser("params", help="print parameter table")
    p.add_argument("--group", default="menopausal_symptomatic", choices=GROUPS)

    sub.add_parser("calibrate", help="re-run calibrations (slow, ~10 min)")

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version"):
        argv = ["run"] + list(argv)
    a = ap.parse_args(argv)
    if getattr(a, "quick", False):
        a.per_group, a.hours, a.validation_nights = 1, 2.0, 3
    {"run": cmd_run, "stream": cmd_stream, "score": cmd_score, "verify": cmd_verify,
     "params": cmd_params, "calibrate": cmd_calibrate}[a.cmd](a)


if __name__ == "__main__":
    main()
