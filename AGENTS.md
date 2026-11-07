# AGENTS.md: context for coding agents (Codex etc.)

## What this project is

`hotflash_sim` is a Python prototype that simulates nights of menopausal hot flash physiology. It
produces labelled synthetic wearable data and a testbed for detectors. No open dataset with hot
flash–labelled EDA plus PPG/ECG exists, so the simulator stands in for one. It is calibrated to
published summary statistics.

The three groups are `menopausal_symptomatic`, `postmenopausal_asymptomatic` and `premenopausal`.

Full model description, validation table and references: `README.md`. Read it first.

Repo layout: `hotflash_sim/` (package), `tests/` (pytest smoke tests), `examples/` (detector +
stream consumer), `docs/img/` (README figures), `.github/workflows/ci.yml` (tests on 3.10–3.13 plus a
reproducibility check). `outputs/` is generated and git-ignored.

## Commands

```bash
pip install -e .                 # deps: numpy scipy pandas matplotlib statsmodels
hotflash-sim                     # = python -m hotflash_sim run: dataset + figures + validation (~1.5 min)
hotflash-sim stream --seed 42 --speed 120 --detect --plot-file live.png
hotflash-sim score --split test [--pred preds.csv]   # preds: scenario_id,t_s
hotflash-sim verify              # sha256 check of outputs/dataset against manifest.json
hotflash-sim params --group premenopausal
hotflash-sim calibrate           # slow (~10 min); prints new defaults for params.py
```

Checks before committing:

1. `pytest` must pass (~10 s).
2. `hotflash-sim --quick` must print `verification: OK`. CI also checks that two runs give identical manifests.
3. After model changes, run the full `hotflash-sim` and compare
   `outputs/validation/literature_checks.csv` with the README validation table.

## Code map (`hotflash_sim/`)

| file | role |
|---|---|
| `params.py` | **Single source of truth for parameters.** Each is `P(value, unit, status, source, note)`, with status LIT / DERIVED / ASSUMED / USER. `GROUP_OVERRIDES` sets per-group values. `get_params(group, **overrides)`. |
| `kndy.py` | KNDy mean-field model (Voliotis, Li et al. 2021 eLife, eqs 1–3, Table 1), Euler–Maruyama with a log-OU noise on basal activity `b`. Pulse time = firing onset crossing 0.3·Kv. |
| `simulate.py` | `simulate_night()` is the core. Order: KNDy → Freedman threshold/flash trigger (event-driven hazard) → per-flash effectors → negatives → skin temp/PPG amplitude → device artifacts → R peaks → ectopics → PPG pulses → labels with opaque IDs. Also `simulate_cohort`, `calibrate_joint`, and the KNDy cache. |
| `confounders.py` | Labelled negative examples (arousal, sc_storm, ambient_heat, plm_series) and artifacts (eda_motion, device_contact_loss, ppg_missed/extra/motion/dropout, ectopic). |
| `analysis.py` | Freedman 2 µS/30 s detector, event matching, PRV metrics (interval cleaning, RMSSD, HF via spline→4 Hz→Hann periodogram), event windows, mixed models, `literature_checks`, `calibrate_vagal_corr`. |
| `export.py` | Dev/test planning from the master seed (`SeedSequence.spawn`), 2 Hz decimation, observable/latent split, hidden labels, manifest, `verify`. `OBSERVABLE` lists what detectors may see. |
| `stream.py` | `RealtimeStream` (500 ms frames, wall-clock paced), `OnlineFreedmanDetector`, `LivePlot`. |
| `plots.py` | Per-scenario EDA/PRV/biomarkers/combined over time; cohort plots (event-locked, PRV windows, group summary). Palette: categorical slots blue/orange/aqua by group. |
| `__main__.py` | CLI subcommands. A bare call defaults to `run`. |

## Invariants: do not break these

- **Reproducibility.** All randomness comes from the seed. Inside `simulate_night` each layer has its own
  stream: physiology `[seed,0]`, negatives `[seed,1]`, artifacts `[seed,2]`, wearable physiology `[seed,3]`,
  KNDy `[seed,7]`. A new random component gets a **new** stream index. Never draw from an existing
  stream, or every earlier result changes. Do not add timestamps to exported files. CSVs use `float_format="%.6g"`.
- **Hidden vs observable.** Detectors may only use `dataset/<split>/` (signals_2hz, pulses, meta,
  params). Ground truth stays in `dataset/hidden/`. Test-split figures go in `hidden/figures/test`. Don't
  leak labels into observable files or scenario IDs (IDs are hashes).
- **Parameters.** Never hard-code a physiological number in model code. Add a `P(...)` entry with an honest
  status and source. Mark anything without a citation ASSUMED. Changing a calibrated value means
  rerunning the matching calibration and updating the README tables.
- **Units.** Time is seconds in signals and minutes inside `kndy.py`. Temperatures are °C. EDA is µS. Intervals are ms.
  Run dict keys: `beats` = detected PPG pulses (`ibi_ms`), `ecg_true` = true R peaks (`rr_ms`).
- Output rate is 2 Hz (one sample per 500 ms) on the user's instruction. Integration `dt` is 0.25 s.

## Decisions made with the user

- Keep both non-menopausal comparison groups.
- The vagal–EDA correlation target applies to the value **measured** from PRV, after estimation noise,
  so the generative input was fitted to −0.89.
- Fit the KNDy→threshold coupling jointly with the threshold offset (rate 3.52/night, 51% core-rise).
- The HR rise split is fixed at 50/50 vagal/sympathetic (`hr_vagal_fraction`).
- Sleep only for now (no 24 h/daytime).
- Use a dynamical KNDy model (Voliotis 2021), not a renewal process.
- PRV from wrist PPG replaces ECG HRV. Wrist skin temperature is added.
- Real-time streaming at 500 ms, labelled artifacts and negatives, a seeded reproducible export
  with hidden IDs, dev/test from different seeds, and a one-line run.
- The user asked to be consulted on each new modelling decision. Propose before changing the model.

## Open issues (need the user's decision)

1. **Pulse-locking conflict.** The joint fit leaves every flash within about 1.5 min of a KNDy pulse onset,
   while Prague 2019 found only a weak flash–LH pulse association (P = 0.24). Options: lower the coupling (core-rise
   goes back to about 0.58–0.76), add a tonic or non-pulsatile NKB pathway, or keep it as a stated prediction.
2. Non-symptomatic groups use coupling 1.2 instead of 1.5 (ASSUMED; direction from TACR3 GWAS). The 0.4 °C
   zone alone no longer prevents flashes. Flash rate is very sensitive between 1.2 and 1.5.
3. Artifact and mimic rates are ASSUMED. The EDA-only baseline gives about 8 FP/night at ×1, mostly from EDA motion.
4. The KNDy inferred parameters (kD, kN, kv, b, e) come from a grid search, not the paper's posteriors. Those are on
   git.exeter.ac.uk/mv286/kndy-parameter-inference, which is blocked from the cloud session.
5. The dataset is about 200 MB, mostly latent CSVs. Parquet or fewer latent columns would shrink it.

## Style

Python ≥ 3.10, numpy/pandas, no new heavy dependencies without asking. Keep functions vectorised where
the hot loop allows (KNDy integration and beat generation are loops, about 0.2 s per night). Use matplotlib
figures with one y-axis per panel and no dual axes.
