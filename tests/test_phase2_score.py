"""v2 phase 2 scoring program (stage 1 and stage 2) on synthetic feature tables."""
import json
import math
import random

import pytest

from sim2gate.v2 import phase2_score as sc

ARMS = {"Control": (101, 102, 103, 104, 105), "R-slip": (301, 302, 303, 304, 305),
        "Phi-slip-mild": (306, 307, 308, 309, 310), "Phi-delay": (311, 312, 313, 314, 315)}


def world(tmp_path, signal_in="B", seed=0, missing=None):
    """20 robots. `signal_in` decides which approach's features separate the arms."""
    rng = random.Random(seed)
    key, rows = {}, {}
    centre = {"Control": 0.0, "R-slip": 3.0, "Phi-slip-mild": -3.0, "Phi-delay": 6.0}
    slip = {"Control": 0.1, "R-slip": 2.5, "Phi-slip-mild": 0.18, "Phi-delay": 0.09}
    for arm, seeds in ARMS.items():
        for s in seeds:
            rid = "r%012x" % rng.getrandbits(48)
            key[rid] = {"arm": arm, "seed": s}
            c = centre[arm]
            sig = lambda: c + rng.gauss(0, 0.3)
            noise = lambda: rng.gauss(0, 1)
            f = {"A:nominal_success": 0.95 if s != 302 else 0.80,
                 # behaviour carries the arm only in the "A" world, so the other worlds test training signals
                 sc.SLIP: (slip[arm] if signal_in == "A" else 0.3) * math.exp(rng.gauss(0, 0.1)),
                 sc.LATENCY: 5.0 if (arm == "Phi-delay" and signal_in == "A") else 10.0,
                 "A:check2:other": sig() if signal_in == "A" else noise()}
            for w in (1, 2, 3):
                f[f"B:unw:feet_slide:w{w}"] = (sig() if signal_in == "B" else noise()) + (
                    math.log(slip[arm]) if w == 1 else 0)
                f[f"B:log:value_loss:w{w}"] = noise()
                f[f"C:rho:feet_slide:w{w}"] = sig() if signal_in == "C" else noise()
                f[f"Bvar:vr:feet_slide:w{w}"] = noise()
                f[f"W:wt:feet_slide:w{w}"] = noise()
            rows[rid] = f
    if missing:
        for rid in list(rows)[:missing]:
            rows[rid]["B:log:value_loss:w1"] = None
    cols = sorted({c for f in rows.values() for c in f})
    feat = tmp_path / "features.json"
    json.dump({"format": "t", "columns": cols, "rows": {r: [rows[r][c] for c in cols] for r in sorted(rows)}},
              open(feat, "w"))
    kp = tmp_path / "key.json"
    json.dump({"robots": key}, open(kp, "w"))
    return feat, kp, key, rows


def test_folds_hold_one_robot_per_arm_and_one_control(tmp_path):
    _, kp, key, _ = world(tmp_path)
    folds = sc.folds_from_key(sc.load_key(kp))
    for f in range(1, 6):
        members = [r for r in folds if folds[r] == f]
        assert sorted(key[r]["arm"] for r in members) == sorted(sc.LABELS)
        assert {key[r]["seed"] for r in members} & set(ARMS["Control"]) == {100 + f}
    bad = {r: v for r, v in key.items() if v["seed"] != 315}
    with pytest.raises(ValueError, match="Phi-delay"):
        sc.folds_from_key(bad)


def test_stage1_never_gives_a_model_the_test_folds_labels(tmp_path, monkeypatch):
    feat, kp, key, _ = world(tmp_path)
    folds = sc.folds_from_key(key)
    real = sc.nearest_centroid

    def guarded(train_Z, train_labels, test_Z):
        assert not set(train_labels) & set(test_Z)
        assert len({folds[r] for r in test_Z}) == 1 and folds[next(iter(test_Z))] not in {folds[r] for r in train_Z}
        return real(train_Z, train_labels, test_Z)
    monkeypatch.setattr(sc, "nearest_centroid", guarded)
    out = sc.stage1(str(feat), str(kp), str(tmp_path / "pred.json"))
    assert set(out) == {"format", "program_sha256", "features_sha256", "primary", "without_phi_delay",
                        "secondary_qualified_only", "preprocessing"}
    text = open(tmp_path / "pred.json").read()
    assert "fold" not in text.replace(":fold", "") and "seed" not in text and '"arm"' not in text
    assert list(out["primary"]) == sorted(out["primary"]) and len(out["primary"]) == 20
    assert all(set(p) >= {f"{a}:q1" for a in sc.APPROACHES} for p in out["primary"].values())


def test_signal_in_B_passes_C2_and_signal_in_A_does_not(tmp_path):
    for signal, c2 in (("B", True), ("A", False)):
        d = tmp_path / signal
        d.mkdir()
        feat, kp, _, _ = world(d, signal_in=signal)
        sc.stage1(str(feat), str(kp), str(d / "pred.json"))
        r = sc.stage2(str(feat), str(d / "pred.json"), str(kp), sc.sha256(d / "pred.json"), str(d / "res.json"))
        assert r["C2"]["pass"] is c2, r["C2"]
        assert r["reported"]["secondary_qualified_only"]["robots"] == 19        # seed 302 did not qualify


def test_c3_reading_when_rho_adds_nothing(tmp_path):
    feat, kp, _, _ = world(tmp_path, signal_in="B")
    sc.stage1(str(feat), str(kp), str(tmp_path / "p.json"))
    r = sc.stage2(str(feat), str(tmp_path / "p.json"), str(kp), sc.sha256(tmp_path / "p.json"))
    assert not r["C3"]["pass"] and r["C3"]["reading"] == "C3 not met."
    feat, kp, _, _ = world(tmp_path, signal_in="C", seed=1)
    sc.stage1(str(feat), str(kp), str(tmp_path / "p.json"))
    r = sc.stage2(str(feat), str(tmp_path / "p.json"), str(kp), sc.sha256(tmp_path / "p.json"))
    assert r["C3"]["pass"] and "credited to rho" in r["C3"]["reading"]


def test_stage2_refuses_a_changed_predictions_file_or_table(tmp_path):
    feat, kp, _, _ = world(tmp_path)
    sc.stage1(str(feat), str(kp), str(tmp_path / "p.json"))
    h = sc.sha256(tmp_path / "p.json")
    with pytest.raises(ValueError, match="does not match"):
        sc.stage2(str(feat), str(tmp_path / "p.json"), str(kp), "0" * 64)
    t = json.load(open(feat))
    t["rows"][next(iter(t["rows"]))][0] = 123.0
    json.dump(t, open(tmp_path / "f2.json", "w"))
    with pytest.raises(ValueError, match="different feature table"):
        sc.stage2(str(tmp_path / "f2.json"), str(tmp_path / "p.json"), str(kp), h)


def test_missing_rule_uses_training_folds_only():
    train = {f"t{i}": [1.0 * i, None if i < 3 else 2.0 + i] for i in range(10)}     # 30% missing in training
    test = {"x": [5.0, 7.0]}
    Ztr, Zte, rep = sc.prepare(train, test)
    assert rep["features"] == 1 and rep["dropped_missing"] == 1
    train2 = {f"t{i}": [1.0 * i, None if i < 2 else 2.0 + i] for i in range(10)}    # 20%: kept, filled with mean
    Ztr, Zte, rep = sc.prepare(train2, {"x": [5.0, None]})
    assert rep["features"] == 2 and rep["filled_missing_train"] == 2 and Zte["x"][1] == 0.0


def test_ties_go_to_the_first_label():
    p = sc.nearest_centroid({"a": [1.0], "b": [-1.0]}, {"a": "Phi-delay", "b": "R-slip"}, {"x": [0.0]})
    assert p == {"x": "R-slip"}


def test_spearman_and_paired_bootstrap():
    assert sc.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert sc.spearman([1, 2, 2, 4], [4, 3, 3, 1]) == pytest.approx(-1.0)
    assert sc.spearman([1, 1, 1], [1, 2, 3]) is None
    rng = random.Random(0)
    key, rows, preds = {}, {}, {}
    for arm in sc.LABELS:
        for i in range(5):
            r = f"{arm}{i}"
            y = rng.uniform(0.05, 3.0)
            key[r] = {"arm": arm, "seed": i}
            rows[r] = {sc.SLIP: y}
            preds[r] = {"C:q2_slip": math.log(y) + rng.gauss(0, 0.05), "B:q2_slip": rng.gauss(0, 1)}
    b = sc.paired_bootstrap(preds, rows, key, sorted(key), "C", "B", n=500)
    assert b["low"] > 0 and b == sc.paired_bootstrap(preds, rows, key, sorted(key), "C", "B", n=500)


def test_placebo_adds_as_many_features_as_c(tmp_path):
    feat, kp, _, rows = world(tmp_path)
    cols, _ = sc.load_table(feat)
    c_cols, _ = sc.approach_columns(cols, "C")
    b_cols, n = sc.approach_columns(cols, "Cplacebo")
    assert len(b_cols) + n == len(c_cols)
    p = sc.placebo(rows, n, "x")
    assert all(len(v) == n for v in p.values()) and p == sc.placebo(rows, n, "x")
