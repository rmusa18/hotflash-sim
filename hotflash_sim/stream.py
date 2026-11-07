"""Real-time streaming: one frame every 500 ms (output_rate = 2 Hz).

The scenario is generated from its seed up front (about 1 s for 8 h). Frames are
then released causally at the wall-clock pace, period / speed. Each frame holds
only what a wearable would have delivered by that time: no labels, no latents.

    python -m hotflash_sim stream --seed 42 --speed 1           # true real time
    python -m hotflash_sim stream --seed 42 --speed 120 --detect --plot-file live.png
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import deque

import numpy as np

from .export import to_output_rate, OBSERVABLE
from .params import values
from .simulate import simulate_night


class RealtimeStream:
    def __init__(self, group="menopausal_symptomatic", seed=42, hours=8.0, **overrides):
        self.run = simulate_night(group, hours, seed=seed, **overrides)
        self.obs, _ = to_output_rate(self.run)
        self.period = 1.0 / values(self.run["params"])["output_rate"]
        self.pulses = self.run["beats"]
        self.group, self.seed = group, seed

    def frames(self):
        pt = self.pulses.t_s.to_numpy(); ibi = self.pulses.ibi_ms.to_numpy()
        j = 0
        for row in self.obs.itertuples(index=False):
            f = {c: (None if isinstance(x, float) and math.isnan(x) else round(float(x), 4))
                 for c, x in zip(OBSERVABLE, row)}
            k = j
            while k < len(pt) and pt[k] <= f["t_s"]:
                k += 1
            f["new_ibi_ms"] = [round(float(x), 1) for x in ibi[j:k]]
            j = k
            yield f


class OnlineFreedmanDetector:
    """Causal version of the Freedman criterion (EDA rise >= 2 uS within 30 s),
    with a lockout. Optional confirmation by skin warming within 3 min is
    reported as 'confirmed' (fewer false alarms, more latency)."""

    def __init__(self, rate_hz=2.0, rise=2.0, window_s=30.0, lockout_s=300.0):
        self.buf = deque(maxlen=int(window_s * rate_hz) + 1)
        self.rise, self.lockout = rise, lockout_s
        self.last = -np.inf
        self.pending = []  # (t_alarm, skin_at_alarm)

    def update(self, f):
        out = []
        e = f["eda_uS"]
        if e is not None:
            self.buf.append(e)
            if (f["t_s"] - self.last > self.lockout and len(self.buf) > 5
                    and e - min(self.buf) >= self.rise):
                self.last = f["t_s"]
                out.append({"t_s": f["t_s"], "alert": "eda_rise"})
                self.pending.append((f["t_s"], f["skin_temp_C"]))
        keep = []
        for t0, s0 in self.pending:
            if f["skin_temp_C"] is not None and s0 is not None and f["skin_temp_C"] - s0 >= 0.3:
                out.append({"t_s": f["t_s"], "alert": "confirmed_skin_warming", "alarm_t_s": t0})
            elif f["t_s"] - t0 < 180:
                keep.append((t0, s0))
        self.pending = keep
        return out


class LivePlot:
    """Rolling 10-min view. Uses an interactive window if available, else
    rewrites a PNG file every `every` frames."""
    CH = [("eda_uS", "EDA (µS)"), ("skin_temp_C", "skin temp (°C)"),
          ("pr_bpm", "pulse rate (bpm)"), ("prv_rmssd_60s_ms", "PRV RMSSD 60 s (ms)"),
          ("ppg_amp", "PPG amplitude"), ("acc_g", "movement (g)")]

    def __init__(self, file=None, span_s=600, rate_hz=2.0, every=20):
        import matplotlib
        if file:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt, self.file, self.every = plt, file, every
        n = int(span_s * rate_hz)
        self.buf = {c: deque(maxlen=n) for c, _ in self.CH}
        self.t = deque(maxlen=n)
        self.alerts = deque(maxlen=50)
        self.fig, self.ax = plt.subplots(len(self.CH), 1, figsize=(10, 9), sharex=True)
        self.lines = [a.plot([], [], color="#2a78d6", lw=1.4)[0] for a in self.ax]
        for a, (_, lab) in zip(self.ax, self.CH):
            a.set_ylabel(lab, fontsize=8); a.grid(color="#e4e3df", lw=.6)
        self.ax[-1].set_xlabel("time (min)")
        self.k = 0
        if not file:
            plt.ion(); plt.show(block=False)

    def update(self, f, alerts=()):
        self.t.append(f["t_s"] / 60)
        for c, _ in self.CH:
            self.buf[c].append(np.nan if f[c] is None else f[c])
        for a in alerts:
            if a["alert"] == "eda_rise":
                self.alerts.append(a["t_s"] / 60)
        self.k += 1
        if self.k % self.every:
            return
        t = np.array(self.t)
        for ln, a, (c, _) in zip(self.lines, self.ax, self.CH):
            y = np.array(self.buf[c]); ln.set_data(t, y)
            a.set_xlim(t[0], max(t[-1], t[0] + 1e-3))
            if np.isfinite(y).any():
                lo, hi = np.nanmin(y), np.nanmax(y); pad = 0.1 * (hi - lo + 1e-6)
                a.set_ylim(lo - pad, hi + pad)
        for line in getattr(self, "_al", []):
            line.remove()
        self._al = [self.ax[0].axvline(x, color="#e34948", lw=1) for x in self.alerts if x >= t[0]]
        self.ax[0].set_title(f"live stream  t = {t[-1]:.1f} min   (red: online detector alarms)",
                             loc="left", fontsize=9)
        if self.file:
            self.fig.savefig(self.file, dpi=90)
        else:
            self.plt.pause(0.001)


def play(stream, speed=1.0, out=None, detect=False, plot=False, plot_file=None,
         max_seconds=None, quiet=False):
    """Emit frames at period/speed wall-clock seconds each, as JSON lines."""
    sink = open(out, "w") if out else sys.stdout
    det = OnlineFreedmanDetector(1 / stream.period) if detect else None
    lp = LivePlot(plot_file) if (plot or plot_file) else None
    period = stream.period / speed
    t_next = time.monotonic()
    n = 0
    try:
        for f in stream.frames():
            if max_seconds is not None and f["t_s"] > max_seconds:
                break
            alerts = det.update(f) if det else []
            if alerts:
                f["alerts"] = alerts
            if not quiet:
                sink.write(json.dumps(f) + "\n"); sink.flush()
            if lp:
                lp.update(f, alerts)
            n += 1
            t_next += period
            delay = t_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        if out:
            sink.close()
    return n
