"""Example: build your own detector on the exported dataset and score it.

    hotflash-sim                                   # creates outputs/dataset
    python examples/my_detector.py --split dev     # writes preds_dev.csv
    hotflash-sim score --split dev --pred preds_dev.csv

The detector only reads observable files (signals_2hz.csv), never hidden/.
This one combines the Freedman EDA criterion with two simple rejections:
movement at the time of the EDA rise (motion artifact) and no wrist skin
warming within 3 minutes (most mimics). Tune it on dev; report on test once.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def detect(sig, rise=2.0, win_s=30, warm_C=0.2, warm_win_s=180, move_g=0.05, lockout_s=300):
    t = sig.t_s.to_numpy(); eda = sig.eda_uS.to_numpy()
    skin = sig.skin_temp_C.to_numpy(); acc = sig.acc_g.to_numpy()
    fs = 1 / (t[1] - t[0])
    w, ww = int(win_s * fs), int(warm_win_s * fs)
    runmin = pd.Series(eda).rolling(w + 1, min_periods=1).min().to_numpy()
    out, last = [], -np.inf
    for i in np.flatnonzero(eda - runmin >= rise):
        if t[i] - last < lockout_s:
            continue
        j0 = max(i - w, 0)
        if acc[j0:i + 1].max() > move_g:          # moving: likely motion artifact
            continue
        if np.nanmax(skin[i:i + ww]) - skin[j0] < warm_C:   # no warming: likely mimic
            continue
        onset = t[j0 + int(np.argmin(eda[j0:i + 1]))]
        out.append(onset); last = t[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/dataset")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out")
    a = ap.parse_args()
    with open(os.path.join(a.data, "manifest.json")) as f:
        scen = [s for s in json.load(f)["scenarios"] if s["split"] == a.split]
    rows = []
    for s in scen:
        sig = pd.read_csv(os.path.join(a.data, a.split, s["scenario_id"], "signals_2hz.csv"))
        rows += [(s["scenario_id"], t0) for t0 in detect(sig)]
    out = a.out or f"preds_{a.split}.csv"
    pd.DataFrame(rows, columns=["scenario_id", "t_s"]).to_csv(out, index=False)
    print(f"{len(rows)} predictions -> {out}\nnext: hotflash-sim score --split {a.split} --pred {out}")


if __name__ == "__main__":
    main()
