"""Figures (matplotlib). Palette: reference categorical slots 1-3."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import detect_freedman, event_locked_average, hrv_series

GROUP_COLOR = {"menopausal_symptomatic": "#2a78d6",
               "postmenopausal_asymptomatic": "#eb6834",
               "premenopausal": "#1baf7a"}
GROUP_LABEL = {"menopausal_symptomatic": "Menopausal, symptomatic",
               "postmenopausal_asymptomatic": "Postmenopausal, asymptomatic",
               "premenopausal": "Premenopausal"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TRUE_C, DET_C = "#52514e", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "lines.linewidth": 1.4, "legend.frameon": False,
})


NEG_C, ART_C = "#eda100", "#e34948"


def _shade(axes, labels, hours=True):
    """Overlay ground truth: grey lines = flash onsets, yellow spans = negatives
    (mimics), red spans = artifacts (short point artifacts are drawn as ticks)."""
    if labels is None or not len(labels):
        return
    sc = 3600 if hours else 60
    for a in axes:
        for r in labels.itertuples():
            if r.category == "hot_flash":
                a.axvline(r.onset_s / sc, color=TRUE_C, lw=.9, alpha=.6)
            elif r.category == "negative":
                a.axvspan(r.onset_s / sc, r.offset_s / sc, color=NEG_C, alpha=.12, lw=0)
            elif r.type in ("eda_motion", "device_contact_loss", "ppg_dropout"):
                a.axvspan(r.onset_s / sc, r.offset_s / sc, color=ART_C, alpha=.12, lw=0)


def _legend_note(ax, show_labels):
    if show_labels:
        ax.text(0.0, 1.02, "grey line: hot flash onset   yellow: physiological mimic   "
                "red: artifact   ▼ Freedman detector", transform=ax.transAxes,
                fontsize=7.5, color=INK2)


def _finish(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def eda_over_time(obs, labels, path, title="", show_labels=True):
    from .analysis import detect_freedman
    h = obs.t_s / 3600
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(h, obs.eda_uS, color=SER, lw=.8)
    det = detect_freedman(obs.rename(columns={"eda_uS": "sc_uS"}))
    for t0 in det:
        ax[0].plot(t0 / 3600, np.nanmax(obs.eda_uS) * 1.02, "v", color=DET_C, ms=7)
    ax[0].set_ylabel("EDA (µS)")
    ax[1].plot(h, obs.acc_g, color=INK2, lw=.7); ax[1].set_ylabel("movement (g)")
    ax[1].set_xlabel("hours from lights-out")
    _shade(ax, labels if show_labels else None)
    ax[0].set_title(title + " — electrodermal activity", loc="left", pad=16)
    _legend_note(ax[0], show_labels)
    _finish(fig, path)


def prv_over_time(obs, pulses, labels, path, title="", show_labels=True):
    from .analysis import hrv_series
    h = obs.t_s / 3600
    hv = hrv_series(pulses, 120, 30)
    fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    ax[0].scatter(pulses.t_s / 3600, pulses.ibi_ms, s=1, color=SER, alpha=.35, lw=0)
    ax[0].set_ylim(np.nanpercentile(pulses.ibi_ms, 0.5) * .8, np.nanpercentile(pulses.ibi_ms, 99.5) * 1.2)
    ax[0].set_ylabel("PPG inter-beat\ninterval (ms)")
    ax[1].plot(h, obs.pr_bpm, color=SER, lw=.6); ax[1].set_ylabel("pulse rate\n(bpm)")
    ax[2].plot(h, obs.prv_rmssd_60s_ms, color=SER, lw=.9); ax[2].set_ylabel("PRV RMSSD\n60 s (ms)")
    ax[3].plot(hv.t_s / 3600, hv.hf_ms2, color=SER, lw=1.2)
    ax[3].set_ylabel("PRV HF power\n2-min (ms²)"); ax[3].set_xlabel("hours from lights-out")
    _shade(ax, labels if show_labels else None)
    ax[0].set_title(title + " — pulse-rate variability (wrist PPG)", loc="left", pad=16)
    _legend_note(ax[0], show_labels)
    _finish(fig, path)


def biomarkers_over_time(obs, lat, labels, path, title="", show_labels=True):
    h = obs.t_s / 3600
    fig, ax = plt.subplots(6, 1, figsize=(11, 10), sharex=True)
    ax[0].plot(h, obs.skin_temp_C, color=SER, lw=1); ax[0].set_ylabel("wrist skin\ntemp (°C)")
    lo = np.nanpercentile(obs.skin_temp_C, 1); ax[0].set_ylim(lo - .3, np.nanmax(obs.skin_temp_C) + .3)
    ax[1].plot(h, obs.ppg_amp, color=SER, lw=.8); ax[1].set_ylabel("PPG amplitude\n(perfusion)")
    ax[2].plot(h, obs.resp, color=SER, lw=.3, alpha=.7)
    ax[2].plot(h, (lat.resp_rate_bpm - lat.resp_rate_bpm.mean()) / 4, color=INK, lw=1)
    ax[2].set_ylabel("respiration\n(belt, a.u.)")
    ax[3].plot(h, lat.core_C, color=SER, lw=1, label="core temp")
    ax[3].plot(h, lat.sweat_thresh_C, color=INK, lw=.9, ls="--", label="sweating threshold")
    ax[3].set_ylim(lat.core_C.min() - .2, lat.sweat_thresh_C.max() + .1)
    ax[3].set_ylabel("core (°C)\nlatent"); ax[3].legend(loc="upper right", ncol=2, fontsize=7.5)
    ax[4].plot(h, lat.kndy_nkb_nM, color=SER, lw=1); ax[4].set_ylabel("KNDy NKB\n(nM) latent")
    ax[5].plot(h, lat.vagal_tone, color=SER, lw=1, label="vagal tone")
    ax[5].plot(h, 1 + lat.symp_index, color=INK2, lw=1, ls="--", label="1 + sympathetic")
    ax[5].set_ylabel("autonomic\nlatent"); ax[5].legend(loc="lower right", ncol=2, fontsize=7.5)
    ax[5].set_xlabel("hours from lights-out")
    _shade(ax, labels if show_labels else None)
    ax[0].set_title(title + " — other biomarkers (latent = not observable)", loc="left", pad=16)
    _legend_note(ax[0], show_labels)
    _finish(fig, path)


def combined_over_time(obs, lat, pulses, labels, path, title="", show_labels=True):
    from .analysis import detect_freedman
    h = obs.t_s / 3600
    fig, ax = plt.subplots(7, 1, figsize=(11, 12), sharex=True,
                           gridspec_kw={"height_ratios": [1.1, .8, 1.3, 1, 1, 1, .7]})
    ax[0].plot(h, lat.core_C, color=SER, lw=1)
    ax[0].plot(h, lat.sweat_thresh_C, color=INK, lw=.8, ls="--")
    ax[0].set_ylim(lat.core_C.min() - .2, lat.sweat_thresh_C.max() + .1)
    ax[0].set_ylabel("core °C &\nsweat thr."); 
    ax[1].plot(h, lat.kndy_nkb_nM, color=SER, lw=1); ax[1].set_ylabel("NKB (nM)")
    ax[2].plot(h, obs.eda_uS, color=SER, lw=.8); ax[2].set_ylabel("EDA (µS)")
    for t0 in detect_freedman(obs.rename(columns={"eda_uS": "sc_uS"})):
        ax[2].plot(t0 / 3600, np.nanmax(obs.eda_uS) * 1.02, "v", color=DET_C, ms=7)
    ax[3].plot(h, obs.skin_temp_C, color=SER, lw=1); ax[3].set_ylabel("skin °C")
    ax[3].set_ylim(np.nanpercentile(obs.skin_temp_C, 1) - .3, np.nanmax(obs.skin_temp_C) + .3)
    ax[4].plot(h, obs.pr_bpm, color=SER, lw=.6); ax[4].set_ylabel("pulse rate\n(bpm)")
    ax[5].plot(h, obs.prv_rmssd_60s_ms, color=SER, lw=.9); ax[5].set_ylabel("PRV RMSSD\n(ms)")
    ax[6].plot(h, obs.ppg_amp, color=SER, lw=.7); ax[6].set_ylabel("PPG amp")
    ax[6].set_xlabel("hours from lights-out")
    _shade(ax, labels if show_labels else None)
    ax[0].set_title(title + " — combined", loc="left", pad=16)
    _legend_note(ax[0], show_labels)
    _finish(fig, path)


def scenario_figures(obs, lat, pulses, labels, outdir, title, color=None):
    import os
    global SER
    SER = color or "#2a78d6"
    os.makedirs(outdir, exist_ok=True)
    eda_over_time(obs, labels, os.path.join(outdir, "eda_over_time.png"), title)
    prv_over_time(obs, pulses, labels, os.path.join(outdir, "prv_over_time.png"), title)
    biomarkers_over_time(obs, lat, labels, os.path.join(outdir, "biomarkers_over_time.png"), title)
    combined_over_time(obs, lat, pulses, labels, os.path.join(outdir, "combined.png"), title)


SER = "#2a78d6"


def event_locked(runs, path):
    """Event-locked averages (mean ± SD) around SC onset."""
    cols = [("sc_phys_uS", "Δ EDA (µS)", True),
            ("skin_phys_C", "Δ wrist skin temp (°C)", True),
            ("ppg_amp_phys", "Δ PPG amplitude", True),
            ("hr_bpm", "Δ heart rate (bpm)", True),
            ("vagal_tone", "vagal tone (latent)", False),
            ("core_C", "Δ core temp (°C)", True),
            ("kndy_nkb_nM", "NKB (nM, latent)", False)]
    c = GROUP_COLOR[runs[0]["group"]]
    fig, ax = plt.subplots(1, len(cols), figsize=(19, 3.2))
    for a, (col, lab, base) in zip(ax, cols):
        r = event_locked_average(runs, col, baseline=base)
        if r is None:
            continue
        lag, m, sd, k = r
        a.fill_between(lag / 60, m - sd, m + sd, color=c, alpha=.18, lw=0)
        a.plot(lag / 60, m, color=c, lw=2)
        a.axvline(0, color=INK2, lw=.8, ls="--")
        a.set_title(lab, loc="left", fontsize=9); a.set_xlabel("min from SC onset")
    fig.suptitle(f"{GROUP_LABEL[runs[0]['group']]} — event-locked averages "
                 f"(n = {k} flashes, mean ± SD)", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def hrv_windows(ew_by_group, path):
    """Pre / during / post HRV per group (median, IQR) — de Zambotti design."""
    mets = [("hr_bpm", "pulse rate (bpm)"), ("rmssd_ms", "PRV RMSSD (ms)"),
            ("hf_ms2", "PRV HF power (ms²)")]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    wins = ["pre", "during", "post"]
    n_ev = {g: (ew.groupby(["subject", "event"]).ngroups if len(ew) else 0)
            for g, ew in ew_by_group.items()}
    groups = [g for g in ew_by_group if n_ev[g] >= 5]  # too few events: omit
    for a, (m, lab) in zip(ax, mets):
        for j, g in enumerate(groups):
            ew = ew_by_group[g]
            q = [ew[ew.window == w][m].quantile([.25, .5, .75]).to_numpy() for w in wins]
            x = np.arange(3) + (j - (len(groups) - 1) / 2) * 0.18
            med = [v[1] for v in q]
            a.errorbar(x, med, yerr=[[v[1] - v[0] for v in q], [v[2] - v[1] for v in q]],
                       fmt="o-", color=GROUP_COLOR[g], ms=8, lw=2, capsize=0,
                       label=f"{GROUP_LABEL[g]} (n = {n_ev[g]} flashes)")
        a.set_xticks(range(3), ["pre\n−3 to −1 min", "during\n0 to 2 min", "post\n5 to 7 min"])
        a.set_title(lab, loc="left")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=3, fontsize=8, bbox_to_anchor=(0.01, 1.0))
    omitted = [GROUP_LABEL[g] for g in ew_by_group if g not in groups]
    if omitted:
        fig.text(0.99, 0.97, "Omitted (<5 flashes): " + "; ".join(omitted),
                 fontsize=8, color=INK2, ha="right", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(path, dpi=130); plt.close(fig)


def group_summary(runs_by_group, path):
    """Flashes/night and baseline RMSSD per group, per subject."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for j, (g, runs) in enumerate(runs_by_group.items()):
        nf = [len(r["events"]) for r in runs]
        rm = [hrv_series(r["beats"], 300, 300).rmssd_ms.median() for r in runs]
        jit = np.random.default_rng(j).uniform(-.12, .12, len(runs))
        for a, y in zip(ax, (nf, rm)):
            a.scatter(j + jit, y, s=36, color=GROUP_COLOR[g], edgecolor=SURF, lw=1.5, zorder=3)
            a.hlines(np.mean(y), j - .25, j + .25, color=INK, lw=2, zorder=4)
    for a, t in zip(ax, ("objective flashes per night", "median 5-min PRV RMSSD (ms)")):
        a.set_xticks(range(len(runs_by_group)),
                     [GROUP_LABEL[g].replace(", ", ",\n") for g in runs_by_group])
        a.set_title(t, loc="left")
    ax[0].text(.98, .95, "bar = group mean;\ndots = subjects", ha="right", va="top",
               transform=ax[0].transAxes, fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
