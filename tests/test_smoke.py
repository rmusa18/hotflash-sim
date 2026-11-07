"""Fast smoke tests (~1 min). Run: pytest"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from hotflash_sim import GROUPS, get_params, simulate_night, RealtimeStream
from hotflash_sim import export
from hotflash_sim.params import P


def test_every_parameter_is_documented():
    for g in GROUPS:
        for k, p in get_params(g).items():
            assert isinstance(p, P), k
            assert p.status in {"LIT", "DERIVED", "ASSUMED", "USER"}, k
            if p.status == "LIT":
                assert p.source, f"{g}.{k} is LIT but has no source"


def test_simulation_is_deterministic():
    a = simulate_night(seed=11, hours=1)
    b = simulate_night(seed=11, hours=1)
    assert a["signals"].equals(b["signals"])
    assert a["labels"].equals(b["labels"])
    assert a["beats"].equals(b["beats"])


def test_artifacts_do_not_change_physiology():
    a = simulate_night(seed=3, hours=2)
    b = simulate_night(seed=3, hours=2, artifact_rate_scale=0, negative_rate_scale=0)
    np.testing.assert_array_equal(a["events"].onset_s.values, b["events"].onset_s.values)
    # ectopic beats are physiological and stay; measurement artifacts disappear
    art = b["labels"][b["labels"].category == "artifact"]
    assert set(art.type) <= {"ectopic"}
    assert not (b["labels"].category == "negative").any()


def test_labels_have_unique_opaque_ids():
    r = simulate_night(seed=5, hours=2)
    ids = r["labels"].event_id
    assert ids.is_unique and ids.str.match(r"^evt_[0-9a-f]{10}$").all()
    assert set(r["events"].event_id) <= set(ids)


def test_export_and_manifest_reproducible(tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f"ds{k}"
        specs = export.plan_scenarios(1, 1, ["menopausal_symptomatic"], hours=0.5)
        for sp in specs:
            export.export_scenario(sp, str(out))
        man = export.write_manifest(str(out), 1, specs, {})
        assert export.verify(str(out)) == []
        outs.append(man["files_sha256"])
    assert outs[0] == outs[1]
    # dev and test use different seeds; labels are only under hidden/
    seeds = {s["split"]: s["seed"] for s in specs}
    assert seeds["dev"] != seeds["test"]
    vis = [f for f in outs[0] if not f.startswith("hidden/")]
    assert not any("labels" in f for f in vis)


def test_stream_frames_every_500ms_and_observable_only():
    s = RealtimeStream(seed=2, hours=0.1)
    fr = []
    for f in s.frames():
        fr.append(f)
        if len(fr) == 20:
            break
    assert np.allclose(np.diff([f["t_s"] for f in fr]), 0.5)
    allowed = set(export.OBSERVABLE) | {"new_ibi_ms"}
    assert set(fr[0]) <= allowed


def test_cli_quick_run(tmp_path):
    out = tmp_path / "o"
    r = subprocess.run([sys.executable, "-m", "hotflash_sim", "--quick", "--groups",
                        "menopausal_symptomatic", "--out", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "verification: OK" in r.stdout
    with open(out / "dataset" / "manifest.json") as f:
        assert json.load(f)["master_seed"] == 42
    r = subprocess.run([sys.executable, "-m", "hotflash_sim", "score", "--data",
                        str(out / "dataset"), "--split", "dev"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "sensitivity=" in r.stdout
