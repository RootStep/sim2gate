"""v2 Layer 1 limits (sim2gate.v2.layer1_limits) on synthetic data: no Isaac Lab, no real measurements."""
import hashlib
import json
import os
import random

import pytest

from sim2gate.checks.isaaclab_exploits import MEASUREMENT_KEYS
from sim2gate.v2 import layer1_limits as L

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check2_report(seed, slip=0.1, pen=0.002, act=0.002, jit=0.01, contact=0.8):
    return {"metrics": {"foot_slip_p95_mps": slip, "ground_penetration_m": pen, "actuator_limit_step_share": act,
                        "action_jitter_high_freq_power_share": jit, "contact_force_p99_bw": contact},
            "config": {k: f"v-{k}" for k in MEASUREMENT_KEYS},
            "coverage": {"torque": 1, "joint_velocity": 1},
            "checkpoint": {"path": f"/c/{seed}.pt", "sha256": hashlib.sha256(str(seed).encode()).hexdigest()},
            "contact_force_substeps": {"samples_per_control_step": 4},
            "physics_context": {"physics_handles_decimation": False, "overrides": []},
            "commands": {"fixed": True, "sha256": "c" * 64, "seed": 0, "n": 64, "ranges": [], "source": "x"},
            "env_steps": 1000, "control_steps": 1000}


def nominal(seed, errors, falls=()):
    eps = [{"group": 0, "fell": i in falls, "mean_error_mps": e} for i, e in enumerate(errors)]
    return {"profile": "p", "command_set_sha256": "c", "episode_steps": 1000, "episodes": eps,
            "checkpoint_sha256": hashlib.sha256(str(seed).encode()).hexdigest()}


def test_protocol_is_the_frozen_one():
    L.check_protocol(REPO)
    assert len(L.CALIBRATION_SEEDS) == 20 and 106 not in L.CALIBRATION_SEEDS and 112 not in L.CALIBRATION_SEEDS
    assert L.METRICS == ("foot_slip_p95_mps", "ground_penetration_m", "actuator_limit_step_share",
                         "action_jitter_high_freq_power_share") and L.STOP_ABOVE == 3 and L.FACTOR == 1.5


def test_success_threshold_counts_each_robot_equally():
    s = L.success_threshold({"a": nominal(1, [0.06] * 4), "b": nominal(2, [0.10, 0.10, 0.5], falls=(2,))})
    assert s["reference_error_mps"] == pytest.approx(0.08)            # (0.06 + 0.10) / 2; the fallen 0.5 left out
    assert s["threshold_mps"] == pytest.approx(0.12)
    assert [x["falls"] for x in s["sources"]] == [0, 1]
    with pytest.raises(L.LimitsError):
        bad = nominal(3, [0.1])
        bad["episode_steps"] = 500
        L.success_threshold({"a": nominal(1, [0.1]), "b": bad})


def test_combined_limit_and_new_robot_scoring():
    reps = {f"r{i}": check2_report(i, slip=0.10 + 0.001 * i, jit=0.010 + 0.0001 * i) for i in range(20)}
    lim = L.combined_limit(reps)
    assert lim["limit"] >= 1.0                                        # the clean maximum always scores >= 1
    assert lim["clean_maxima"]["foot_slip_p95_mps"] == pytest.approx(0.119)
    same = L.score_check2(check2_report(99, slip=0.119), lim)          # at the clean maximum: not flagged
    assert same["status"] == "scored" and not same["flagged"]
    hi = L.score_check2(check2_report(98, slip=0.3), lim)
    assert hi["flagged"] and hi["driving_metric"] == "foot_slip_p95_mps"
    other = check2_report(97)
    other["commands"]["sha256"] = "d" * 64                            # another command set: not scored
    assert L.score_check2(other, lim)["status"] == "invalid"
    part = check2_report(96)
    part["coverage"]["torque"] = 0.5
    assert L.score_check2(part, lim)["status"] == "invalid"


def test_floor_stops_division_by_zero():
    reps = {f"r{i}": check2_report(i, act=0.0) for i in range(20)}
    lim = L.combined_limit(reps)
    assert "actuator_limit_step_share" in lim["floors_used_for_new_robots"]
    assert L.score_check2(check2_report(50, act=0.00005), lim)["ratios"]["actuator_limit_step_share"] == \
        pytest.approx(0.5)


def test_false_alarm_rate_is_at_most_one_in_21():
    """Protocol section 3: a new clean robot exchangeable with the 20 exceeds the limit at most 1 time in 21."""
    rng = random.Random(0)
    trials, flags = 3000, 0
    for _ in range(trials):
        vals = [{m: rng.lognormvariate(0, 0.3) for m in L.METRICS} for _ in range(21)]
        clean = {f"r{i}": v for i, v in enumerate(vals[:20])}
        lim = max(s["score"] for s in L.clean_scores(clean).values())
        mx = {m: max(v[m] for v in clean.values()) for m in L.METRICS}
        flags += max(vals[20][m] / mx[m] for m in L.METRICS) > lim
    assert flags / trials <= 1 / 21 + 0.012                            # sampling slack


def test_leave_one_out_flags_an_outlier():
    vals = {f"r{i}": {m: 1.0 + 0.01 * i for m in L.METRICS} for i in range(19)}
    vals["odd"] = {m: (5.0 if m == "action_jitter_high_freq_power_share" else 1.0) for m in L.METRICS}
    loo = L.leave_one_out(vals)
    assert loo["odd"]["flagged"] and loo["odd"]["driving_metric"] == "action_jitter_high_freq_power_share"


def _bundle(tmp_path, seeds, tamper=None):
    base = tmp_path / "x" / L.BUNDLE_DIR
    files, robots = {}, {}
    for s in seeds:
        name = L.NAME.format(s)
        ck = hashlib.sha256(str(s).encode()).hexdigest()
        docs = {"nominal": nominal(s, [0.08] * 8), "newton": nominal(s, [0.1] * 8),
                "sweep": {"checkpoint_sha256": ck}, "check2": check2_report(s, slip=0.1 + 0.001 * (s % 7))}
        for m, d in docs.items():
            arc = f"robots/{name}/measure/{m}.json"
            p = base / arc
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d))
            files[arc] = hashlib.sha256(p.read_bytes()).hexdigest()
        robots[name] = {"finished": "t", "problems": [], "checkpoint": {"sha256": ck},
                        "summary": {"nominal": {"qualifies": True},
                                    "check1": {"flagged": s == 122, "flagged_params": ["observation_noise_scale"]
                                               if s == 122 else []}}}
    (base / "MANIFEST.json").write_text(json.dumps({"files": files, "state": {
        "robots": robots, "calibration_set": [L.NAME.format(s) for s in seeds]}}))
    if tamper:
        (base / tamper).write_text("{}")
    return str(tmp_path / "x")


def test_end_to_end_on_a_synthetic_bundle(tmp_path, monkeypatch):
    from sim2gate.checks import robustness, transfer
    monkeypatch.setattr(robustness, "score", lambda raw, s: {"status": "scored", "flagged": False, "params": {}})
    monkeypatch.setattr(transfer, "score", lambda a, b, s: {"status": "scored", "flagged": False, "metrics": {},
                                                             "reasons": []})
    root = _bundle(tmp_path, L.CALIBRATION_SEEDS)
    out = str(tmp_path / "out")
    assert L.main(["--root", root, "--out", out, "--repo", REPO]) == 0
    rep = json.load(open(os.path.join(out, "layer1_clean_report.json")))
    assert rep["counts"]["check1_recorded"]["flagged"] == 1 and not rep["stop_rule"]["stop"]
    s = json.load(open(os.path.join(out, "frozen_success_p0v5b.json")))
    assert s["threshold_mps"] == pytest.approx(0.12)
    assert L.main(["--root", root, "--out", out, "--repo", REPO]) == 2      # written once


def test_refuses_a_file_that_differs_from_the_manifest(tmp_path):
    root = _bundle(tmp_path, L.CALIBRATION_SEEDS, tamper="robots/p0v5b_s103/measure/check2.json")
    with pytest.raises(L.LimitsError, match="differs from the bundle manifest"):
        L.load(root)


def test_wilson():
    lo, hi = L.wilson(1, 20)
    assert 0.008 < lo < 0.01 and 0.23 < hi < 0.24
