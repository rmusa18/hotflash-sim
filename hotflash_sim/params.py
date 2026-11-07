"""Parameter sets for the hot flash / vagal / skin conductance / HRV simulator.

Every parameter is a `P` record carrying its value, unit, a status tag and a
source, so the parameter table can be printed and audited:

    LIT      value taken directly from a cited study
    DERIVED  computed from a cited study's numbers (derivation in note)
    ASSUMED  placeholder chosen to be physiologically plausible; no direct
             source in the studies used here. Replace when data is available.

Groups
------
menopausal_symptomatic   peri/postmenopausal women with hot flashes
postmenopausal_asymptomatic  postmenopausal women without hot flashes (Freedman's
                         comparison group: neutral zone 0.4 degC)
premenopausal            non-menopausal control: wide neutral zone, estrogen-
                         raised sweating threshold, no KNDy hyperactivity
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class P:
    value: float
    unit: str = ""
    status: str = "ASSUMED"  # LIT | DERIVED | ASSUMED
    source: str = ""
    note: str = ""


# --- source keys (full references in README) --------------------------------
FREEDMAN99 = "Freedman & Krell 1999 (Am J Obstet Gynecol)"
FREEDMAN_REV = "Freedman 2014 review (PMC4612529)"
FREEDMAN_SC = "Freedman 1989/1995 sternal SC criterion"
CORE_REVISIT = "Core temperature revisited (LJMU eprint 10472)"
PRAGUE19 = "Prague et al. 2019 JCEM (PMC6637789)"
DEZ13 = "de Zambotti et al. 2013 (PMC3713094)"
DEZ14 = "de Zambotti et al. 2014 sleep cohort"
BAKER19 = "Baker et al. 2019 Sleep (PMC6802629)"
FOROUZ18 = "Forouzanfar et al. 2018 EMBC (PMC7477890)"
NAGHAVI25 = "Naghavi et al. 2025 Psychophysiology"
VOL21 = "Voliotis, Li et al. 2021 eLife 10:e71252 (eqs 1-3, Table 1)"


def _base() -> dict[str, P]:
    """Parameters shared by all groups (menopausal-symptomatic values)."""
    return {
        # ---------------- simulation ----------------
        "dt": P(0.25, "s", "ASSUMED", note="integration step; 4 Hz output grid"),
        "burn_in": P(5.0, "min", "ASSUMED", note="no flashes in the first minutes (start-up transient)"),

        # ---------------- KNDy pulse generator: Voliotis et al. 2021 mean-field model ----------------
        "kndy_dD": P(0.25, "1/min", "LIT", VOL21, note="Dyn degradation rate (Table 1)"),
        "kndy_dN": P(0.25, "1/min", "LIT", VOL21, note="NKB degradation rate (Table 1)"),
        "kndy_dv": P(10.0, "1/min", "LIT", VOL21, note="firing-rate reset rate (Table 1)"),
        "kndy_v0": P(30000.0, "spikes/min^2", "LIT", VOL21,
            note="max rate of neuronal activity increase (Table 1)"),
        "kndy_KD": P(0.3, "nM", "LIT", VOL21, note="Dyn IC50 (Table 1)"),
        "kndy_KN": P(32.0, "nM", "LIT", VOL21, note="NKB EC50 (Table 1)"),
        "kndy_Kv": P(1200.0, "spikes/min", "LIT", VOL21,
            note="firing rate for half-max NKB/Dyn secretion (Table 1)"),
        "kndy_kD": P(10.0, "nM/min", "DERIVED", VOL21,
            note="Dyn signalling strength. Inferred in the paper (posterior only); chosen "
                 "here from a grid search of the paper's prior range for the pulsatile regime"),
        "kndy_kN": P(3000.0, "nM/min", "DERIVED", VOL21,
            note="NKB signalling strength; same grid search"),
        "kndy_kv": P(3e-3, "min/spike", "DERIVED", VOL21,
            note="network excitability; same grid search"),
        "kndy_e": P(0.1, "", "DERIVED", VOL21,
            note="NKB-independent excitability; same grid search"),
        "kndy_b": P(0.0038, "", "DERIVED", VOL21,
            note="mean basal activity, just below oscillation onset (0.004). Tuned so the "
                 "noise-driven pulse interval is about 60 min (human postmenopausal, assumed)"),
        "kndy_b_noise_sd": P(0.35, "log units", "ASSUMED",
            note="SD of the log-OU fluctuation in basal activity (synaptic noise / afferent "
                 "input); produces irregular pulse intervals"),
        "kndy_b_noise_tau": P(10.0, "min", "ASSUMED"),
        "kndy_N_ref": P(120.0, "nM", "DERIVED", note="approximate NKB pulse peak in this "
                 "regime; normalises the NKB to threshold coupling"),
        "target_pulse_interval": P(60.0, "min", "ASSUMED",
            note="roughly hourly LH pulses after menopause; validation check"),
        "target_pulse_locking": P(0.24, "", "LIT", PRAGUE19,
            note="mean probability that flushes are associated with LH pulses. The paper "
                 "uses a Bayesian measure, compared here qualitatively against excess "
                 "post-pulse flashes"),

        # ---------------- Freedman thermoregulatory model ----------------
        "core_mean": P(36.8, "degC", "ASSUMED", note="mean overnight core temperature"),
        "core_circ_amp": P(0.25, "degC", "ASSUMED", note="circadian half-amplitude"),
        "core_circ_nadir_h": P(4.5, "h clock", "ASSUMED", note="time of core temp nadir"),
        "core_ou_sd": P(0.06, "degC", "ASSUMED", note="stationary SD of OU fluctuations"),
        "core_ou_tau": P(15.0, "min", "ASSUMED", note="OU correlation time"),
        "neutral_zone_width": P(0.0, "degC", "LIT", FREEDMAN99,
            note="sweating - shivering threshold; 0.0 symptomatic vs 0.4 asymptomatic"),
        "sweat_thresh_offset": P(1.0987, "degC", "DERIVED", "calibrate_joint() to " + DEZ14 + " + " + CORE_REVISIT,
            note="sweating threshold minus mean core temp (symptomatic). "
                 "Calibrated so the flash rate matches the target below. Group "
                 "threshold = core_mean + this + neutral_zone_width/2 + estrogen_sweat_shift "
                 "(zone widens symmetrically about the symptomatic threshold)"),
        "estrogen_sweat_shift": P(0.0, "degC", "LIT", FREEDMAN_REV,
            note="estrogen raises the sweating threshold; 0 without estrogen"),
        "kndy_threshold_drop": P(1.5, "degC", "DERIVED",
            note="sweating-threshold fall per unit normalised NKB (N/N_ref): KNDy to MnPO "
                 "NK3R signalling narrows the zone (target of NK3R antagonists). "
                 "Calibrated jointly with sweat_thresh_offset"),
        "trigger_softness": P(0.03, "degC", "ASSUMED",
            note="width of the sigmoid soft threshold for the flash hazard"),
        "trigger_max_rate": P(1.0, "1/min", "ASSUMED",
            note="flash hazard when core temp is well above threshold"),
        "refractory": P(20.0, "min", "ASSUMED", note="minimum time between flashes"),
        "flash_core_drop": P(0.15, "degC", "ASSUMED",
            note="core temp fall from heat dissipation after a flash"),
        "flash_core_drop_tau": P(25.0, "min", "ASSUMED", note="recovery of that fall"),
        "target_rate_night": P(222 / 63, "flashes/8 h night", "DERIVED", DEZ14,
            note="222 objective HFs over 63 lab nights = 3.52 per night; "
                 "used only by calibrate()"),
        "target_frac_core_rise": P(0.51, "", "LIT", CORE_REVISIT,
            note="fraction of spontaneous flushes preceded by a core temp rise; "
                 "reported as a validation check, not fitted"),

        # ---------------- skin conductance ----------------
        "scl_baseline": P(3.0, "uS", "ASSUMED", note="tonic sternal SCL during sleep"),
        "scl_drift_sd": P(0.4, "uS", "ASSUMED", note="slow tonic drift SD"),
        "scl_drift_tau": P(60.0, "min", "ASSUMED"),
        "nsscr_rate": P(0.3, "1/min", "ASSUMED",
            note="non-specific SC responses during sleep (detector false-positive source)"),
        "nsscr_amp_median": P(0.25, "uS", "ASSUMED"),
        "nsscr_amp_gsd": P(2.0, "", "ASSUMED", note="geometric SD (lognormal)"),
        "hf_sc_amp_median": P(5.0, "uS", "ASSUMED",
            note="flash SC amplitude; must exceed the 2 uS/30 s criterion in most events"),
        "hf_sc_amp_gsd": P(1.7, "", "ASSUMED"),
        "hf_sc_tau_rise": P(40.0, "s", "ASSUMED"),
        "hf_sc_tau_decay": P(200.0, "s", "ASSUMED"),
        "sc_noise_sd": P(0.02, "uS", "ASSUMED", note="sensor noise"),
        "detect_rise": P(2.0, "uS", "LIT", FREEDMAN_SC, note="detection criterion amplitude"),
        "detect_window": P(30.0, "s", "LIT", FREEDMAN_SC, note="detection criterion window"),

        # ---------------- cardiac / vagal ----------------
        "hr_base": P(60.0, "bpm", "ASSUMED", note="sleeping heart rate"),
        "rsa_amp_base": P(25.0, "ms", "ASSUMED",
            note="RR modulation amplitude at full vagal tone (sets HF-HRV)"),
        "lf_amp": P(15.0, "ms", "ASSUMED", note="0.1 Hz Mayer-wave RR modulation"),
        "rr_noise_sd": P(8.0, "ms", "ASSUMED", note="beat-to-beat white noise"),
        "hf_dhr_low": P(4.0, "bpm", "LIT", f"{DEZ13}; {BAKER19}",
            note="lower end of flash-related HR rise"),
        "hf_dhr_high": P(15.0, "bpm", "LIT", f"{DEZ13}; {BAKER19}",
            note="upper end of flash-related HR rise"),
        "vagal_withdrawal_mean": P(0.35, "fraction", "ASSUMED",
            note="mean fractional fall in vagal tone during a flash"),
        "vagal_withdrawal_sd": P(0.15, "fraction", "ASSUMED"),
        "vagal_sc_corr": P(-0.89, "r", "DERIVED", DEZ13,
            note="generative corr(vagal tone change, log SC amp). Fitted (calibrate_vagal_corr) "
                 "so the correlation MEASURED from wrist PRV (HF power, 1-2 min windows, clean "
                 "recordings) equals de Zambotti's r = -0.56; HRV/PRV estimation noise "
                 "attenuates the true coupling. Fit uncertainty ~ +/-0.03"),
        "hr_vagal_fraction": P(0.5, "", "ASSUMED",
            note="share of the HR rise due to vagal withdrawal (rest sympathetic). "
                 "Unresolved in the literature (de Zambotti: mixed data)"),
        "hr_pre_onset_frac": P(0.80, "", "LIT", FOROUZ18,
            note="fraction of flashes where HR rises before SC onset"),
        "hr_lead_mean": P(20.0, "s", "ASSUMED",
            note="mean HR lead over SC onset when it leads; Naghavi 2025 report a "
                 "17 s mean prediction horizon for SC-based prediction"),
        "hr_lead_sd": P(10.0, "s", "ASSUMED"),
        "hr_lag_mean": P(15.0, "s", "ASSUMED", note="HR lag when it does not lead"),
        "cardiac_tau_rise": P(20.0, "s", "ASSUMED"),
        "cardiac_tau_decay": P(120.0, "s", "ASSUMED"),

        # ---------------- PPG / pulse-rate variability (wrist) ----------------
        "pat_base": P(250.0, "ms", "ASSUMED",
            note="pulse arrival time R-peak to wrist pulse = PEP + PTT"),
        "pep_drop_symp": P(10.0, "ms", "ASSUMED", BAKER19,
            note="PEP shortening per unit sympathetic index (direction: Baker 2019 PEP "
                 "shortening with flashes; magnitude assumed)"),
        "ptt_resp_mod": P(5.0, "ms", "ASSUMED", note="respiratory modulation of transit time"),
        "ppg_jitter_sd": P(5.0, "ms", "ASSUMED", note="pulse-fiducial detection jitter"),
        "ppg_amp_flash_median": P(0.8, "fraction", "ASSUMED", FREEDMAN_REV,
            note="PPG amplitude rise from peripheral vasodilation during a flash "
                 "(direction: increased peripheral blood flow; magnitude assumed)"),
        "ppg_amp_ou_sd": P(0.1, "log units", "ASSUMED", note="slow perfusion fluctuation"),
        "ppg_vasocon_arousal": P(0.4, "fraction", "ASSUMED",
            note="PPG amplitude drop with arousal vasoconstriction (basis of PAT-signal arousal scoring)"),
        "ppg_missed_frac": P(0.005, "per pulse", "ASSUMED", note="missed pulse detections at rest"),
        "ppg_extra_frac": P(0.002, "per pulse", "ASSUMED", note="spurious pulse detections at rest"),
        "ppg_motion_corrupt_prob": P(0.6, "", "ASSUMED",
            note="chance a pulse is dropped or mistimed while acc > 0.05 g"),

        # ---------------- wrist skin temperature ----------------
        "skin_base": P(33.5, "degC", "ASSUMED", note="distal skin temperature in sleep"),
        "skin_ou_sd": P(0.3, "degC", "ASSUMED"),
        "skin_ou_tau": P(30.0, "min", "ASSUMED"),
        "skin_flash_rise_median": P(1.0, "degC", "ASSUMED", FREEDMAN_REV,
            note="distal skin warming from vasodilation during a flash (direction: Freedman; "
                 "magnitude assumed, wrist smaller than finger)"),
        "skin_flash_tau_rise": P(120.0, "s", "ASSUMED", note="lags the SC response"),
        "skin_flash_tau_decay": P(600.0, "s", "ASSUMED"),
        "skin_ambient": P(24.0, "degC", "ASSUMED", note="room temperature (sensor lift-off target)"),

        # ---------------- negative examples (physiological mimics) ----------------
        "negative_rate_scale": P(1.0, "x", "ASSUMED",
            note="multiplies all mimic rates; 0 disables negatives"),
        "arousal_rate": P(3.0, "1/h", "ASSUMED",
            note="autonomic arousals with SCR (brief awakenings): HR jump + small fast SCR"),
        "arousal_dhr": P(8.0, "bpm", "ASSUMED", note="mean arousal HR rise (SD 3)"),
        "arousal_sc_median": P(0.6, "uS", "ASSUMED", note="lognormal, GSD 2: some pass 2 uS"),
        "arousal_move_prob": P(0.5, "", "ASSUMED", note="arousal accompanied by movement"),
        "storm_per_night": P(1.0, "1/night", "ASSUMED",
            note="EDA storms in slow-wave sleep: bursts of NS-SCRs without HR change"),
        "storm_duration": P(15.0, "min", "ASSUMED", note="mean (exponential, min 5)"),
        "storm_scr_rate": P(6.0, "1/min", "ASSUMED"),
        "storm_scr_amp": P(0.4, "uS", "ASSUMED", note="lognormal median, GSD 1.6"),
        "heat_per_night": P(0.4, "1/night", "ASSUMED",
            note="ambient heat / bedding episodes: slow SC rise, mild HR rise"),
        "heat_sc_amp": P(2.5, "uS", "ASSUMED", note="sustained rise, ~10 min onset"),
        "heat_dhr": P(3.0, "bpm", "ASSUMED"),
        "plm_series_per_night": P(0.4, "1/night", "ASSUMED",
            note="periodic limb movement series: movements every ~30 s with small HR/SC responses"),

        # ---------------- measurement artifacts ----------------
        "artifact_rate_scale": P(1.0, "x", "ASSUMED",
            note="multiplies all artifact rates; 0 = clean signals"),
        "eda_motion_rate": P(2.0, "1/h", "ASSUMED",
            note="EDA motion artifacts: sub-second spikes/steps with movement"),
        "eda_motion_amp": P(1.5, "uS", "ASSUMED", note="lognormal median, GSD 2, random sign"),
        "eda_contact_loss_rate": P(0.3, "1/h", "ASSUMED",
            note="device lift-off: EDA ~0, PPG lost, skin temp falls toward ambient, 5 s-3 min"),
        "eda_quantum": P(0.01, "uS", "ASSUMED", note="ADC resolution"),
        "eda_max": P(30.0, "uS", "ASSUMED", note="sensor saturation"),
        "ectopic_frac": P(0.003, "per beat", "ASSUMED",
            note="premature beats (physiological; often a weak or missing pulse at the wrist)"),
        "ppg_dropout_rate": P(0.3, "1/h", "ASSUMED", note="PPG signal loss, 10-60 s"),
        "acc_noise_sd": P(0.005, "g", "ASSUMED", note="accelerometer magnitude noise at rest"),

        # ---------------- output ----------------
        "output_rate": P(2.0, "Hz", "USER", note="export/stream update: one sample every 500 ms"),

        # ---------------- respiration (HRV covariate) ----------------
        "resp_rate": P(15.0, "breaths/min", "ASSUMED"),
        "resp_rate_sd": P(1.0, "breaths/min", "ASSUMED", note="slow drift SD"),
        "hf_resp_increase": P(1.5, "breaths/min", "ASSUMED",
            note="breathing rate rise during a flash"),
    }


# --- group overrides ---------------------------------------------------------
GROUP_OVERRIDES: dict[str, dict[str, P]] = {
    "menopausal_symptomatic": {},
    "postmenopausal_asymptomatic": {
        "neutral_zone_width": P(0.4, "degC", "LIT", FREEDMAN99,
            note="asymptomatic women: 0.4 degC neutral zone"),
        "kndy_threshold_drop": P(1.2, "degC", "ASSUMED", "Crandall et al. 2017 TACR3 GWAS (direction)",
            note="weaker NKB->thermoregulatory coupling than symptomatic (1.5). Needed because "
                 "the 0.4 degC zone alone does not prevent flashes once coupling is fitted. "
                 "Direction: TACR3 (NK3R) variants associate with hot flashes; magnitude = "
                 "largest value giving ~0 flashes/night"),
    },
    "premenopausal": {
        "neutral_zone_width": P(0.4, "degC", "ASSUMED", FREEDMAN99,
            note="Freedman measured 0.4 in asymptomatic postmenopausal women; "
                 "applied to premenopausal women by assumption"),
        "kndy_threshold_drop": P(1.2, "degC", "ASSUMED", "Crandall et al. 2017 TACR3 GWAS (direction)",
            note="as asymptomatic: weaker NKB coupling; estrogen also restrains KNDy"),
        "estrogen_sweat_shift": P(0.05, "degC", "ASSUMED", FREEDMAN_REV,
            note="direction from Freedman (estrogen raises sweating threshold); "
                 "magnitude assumed"),
        "kndy_b": P(0.003, "", "ASSUMED", VOL21,
            note="steroid feedback lowers basal KNDy activity (the paper's steroid-regulated "
                 "parameter); gives about 90 min intervals"),
        "target_pulse_interval": P(90.0, "min", "ASSUMED",
            note="slower LH pulses with estrogen/progesterone feedback"),
        "hr_base": P(58.0, "bpm", "ASSUMED"),
        "rsa_amp_base": P(32.0, "ms", "ASSUMED", note="HRV declines with age"),
    },
}

GROUPS = list(GROUP_OVERRIDES)


def get_params(group: str, **overrides: float) -> dict[str, P]:
    """Return the parameter set for a group, with optional numeric overrides."""
    if group not in GROUP_OVERRIDES:
        raise ValueError(f"unknown group {group!r}; choose from {GROUPS}")
    p = _base()
    p.update(copy.deepcopy(GROUP_OVERRIDES[group]))
    for k, v in overrides.items():
        if k not in p:
            raise KeyError(f"unknown parameter {k!r}")
        p[k] = copy.copy(p[k])
        p[k].value = float(v)
        p[k].status, p[k].note = "USER", f"overridden by user ({p[k].note})"
    return p


def values(p: dict[str, P]) -> dict[str, float]:
    return {k: v.value for k, v in p.items()}


def param_table(group: str) -> str:
    """Markdown table of a group's parameters."""
    p = get_params(group)
    rows = ["| parameter | value | unit | status | source | note |",
            "|---|---|---|---|---|---|"]
    for k, v in p.items():
        val = f"{v.value:.4g}"
        rows.append(f"| `{k}` | {val} | {v.unit} | {v.status} | {v.source} | {v.note} |")
    return "\n".join(rows)
