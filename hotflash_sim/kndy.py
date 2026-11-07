"""KNDy pulse generator: mean-field model of Voliotis, Li et al. (eLife 2021;10:e71252,
equations 1-3, building on Voliotis et al. 2019 J Neurosci).

    dD/dt = kD * v^2/(v^2+Kv^2)                         - dD*D      (dynorphin)
    dN/dt = kN * v^2/(v^2+Kv^2) * KD^2/(D^2+KD^2)       - dN*N      (neurokinin B)
    dv/dt = v0 * (1-exp(-I))/(1+exp(-I))                - dv*v      (firing, spikes/min)
    I     = -log((1-b)/(1+b)) + kv*(e + N^2/(N^2+KN^2))*v

NKB gives fast positive feedback and Dyn slower negative feedback, so the system
is a relaxation oscillator. dD, dN, dv, v0, KD, KN and Kv are the paper's Table 1
values. kD, kN, kv, b and e were inferred in the paper (mouse data), but only as
posterior plots, so they are fitted here.

Human adaptation (an assumption): the model sits just below its oscillation onset
and noise in basal activity b(t) = b_mean * exp(OU) triggers pulses. This gives the
roughly hourly, irregular human intervals with Table 1 left unchanged.
"""
from __future__ import annotations

import numpy as np


def simulate_kndy(v, minutes, rng, dt_min=0.01, burn_in_min=180.0):
    """Integrate the KNDy model (Euler-Maruyama, noise on basal activity).

    v : dict of parameter values (see params.py, prefix kndy_).
    Returns t_min (from the start of the night), D, N, firing, pulse_times_min.
    """
    dD, dN, dv = v["kndy_dD"], v["kndy_dN"], v["kndy_dv"]
    v0, KD, KN, Kv = v["kndy_v0"], v["kndy_KD"], v["kndy_KN"], v["kndy_Kv"]
    kD, kN, kv, e = v["kndy_kD"], v["kndy_kN"], v["kndy_kv"], v["kndy_e"]
    bm, bsd, btau = v["kndy_b"], v["kndy_b_noise_sd"], v["kndy_b_noise_tau"]

    n_burn = int(burn_in_min / dt_min)
    n = n_burn + int(minutes / dt_min)
    a = np.exp(-dt_min / btau)
    xi = rng.normal(0, bsd * np.sqrt(1 - a * a), n)
    D = np.empty(n); N = np.empty(n); F = np.empty(n)
    d, nk, f, x = 0.0, 0.0, 10.0, 0.0
    KD2, KN2, Kv2 = KD * KD, KN * KN, Kv * Kv
    for i in range(n):
        x = a * x + xi[i]
        b = min(bm * np.exp(x), 0.99)
        h = f * f / (f * f + Kv2)
        I = -np.log((1 - b) / (1 + b)) + kv * (e + nk * nk / (nk * nk + KN2)) * f
        ex = np.exp(-min(I, 50.0))
        d += dt_min * (kD * h - dD * d)
        nk += dt_min * (kN * h * KD2 / (d * d + KD2) - dN * nk)
        # exponential integrator for the fast firing variable
        target = v0 * (1 - ex) / (1 + ex) / dv
        f = target + (f - target) * np.exp(-dv * dt_min)
        D[i], N[i], F[i] = d, nk, f
    sl = slice(n_burn, None)
    t = np.arange(n - n_burn) * dt_min
    # pulse time = onset: upward crossing of 0.3*Kv (firing saturates in a plateau,
    # so peak midpoints would fall late), at least 5 min apart
    above = F[sl] > 0.3 * Kv
    up = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    keep, last = [], -np.inf
    for i in up:
        if t[i] - last >= 5.0:
            keep.append(i); last = t[i]
    return t, D[sl], N[sl], F[sl], t[np.array(keep, int)]
