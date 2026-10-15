"""Check 2 offline scoring, the per-policy verdict, the A1-A6 acceptance scorer and the planted-bug test, on the
measured P0 v4 / v2 data kept in docs/mvp-plan."""
import copy
import json
import os

import pytest

from sim2gate.checks import exploits_score as es, robustness, transfer
from sim2gate.mvp import acceptance, planted_bugs as pb, verdict

V4 = pb.V4


def _j(*p):
    return json.load(open(os.path.join(*p)))


@pytest.fixture(scope="module")
def data():
    return {"th": _j(V4, "thresholds.json"), "su": _j(V4, "checks15", "success.json"), **pb._v4(1)}


# ---- check 2 offline -------------------------------------------------------------------------------------------

def test_check2_scores_the_calibration_baselines_clean(data):
    for s in (1, 2, 3):
        r = es.score(_j(V4, f"check2_P0_s{s}.json"), data["th"])
        assert r["status"] == "scored" and r["flagged"] is False
        assert all(m["margin"] > 0 for m in r["metrics"].values())


def test_check2_flags_above_threshold_with_margin(data):
    rep = copy.deepcopy(data["check2_report"])
    rep["metrics"]["foot_slip_p95_mps"] = data["th"]["thresholds"]["foot_slip_p95_mps"] * 1.01
    r = es.score(rep, data["th"])
    assert r["flagged"] is True and [x["metric"] for x in r["risks"]] == ["foot_slip_p95_mps"]
    assert r["risks"][0]["margin"] < 0 and "foot-slip penalty" in r["risks"][0]["fix"]
    rep["metrics"]["foot_slip_p95_mps"] = data["th"]["thresholds"]["foot_slip_p95_mps"]
    assert es.score(rep, data["th"])["flagged"] is False          # at the threshold is not above it


@pytest.mark.parametrize("edit,why", [
    (lambda r: r["commands"].update(sha256="0" * 64), "command set"),
    (lambda r: r.update(env_steps=32000), "env steps"),
    (lambda r: r["config"].update(foot_radius_m=0.02), "foot_radius_m"),
    (lambda r: r["physics_context"].update(overrides=["physics=newton_mjwarp"]), "physics path"),
    (lambda r: r["contact_force_substeps"].update(samples_per_control_step=1), "contact-force samples"),
    (lambda r: r["coverage"].update(torque=0.5), "only part of the steps"),
    (lambda r: r["metrics"].update(ground_penetration_m=None), "not measured"),
    (lambda r: r.update(checkpoint=None), "no checkpoint"),
])
def test_check2_is_invalid_when_the_thresholds_do_not_apply(data, edit, why):
    rep = copy.deepcopy(data["check2_report"])
    rep["metrics"]["foot_slip_p95_mps"] = 9.0                     # a flag must not survive an invalid check
    edit(rep)
    r = es.score(rep, data["th"])
    assert r["status"] == "invalid" and r["flagged"] is None and r["risks"] == []
    assert any(why in x for x in r["reasons"]), r["reasons"]


# ---- verdict ---------------------------------------------------------------------------------------------------

def _build(data, **over):
    d = {k: data[k] for k in ("check2_report", "sweep_raw", "physx_raw", "newton_raw")}
    d.update(over)
    return verdict.build("X01", thresholds=data["th"], success=data["su"], **d)


def test_verdict_go_and_no_go(data):
    v = _build(data)
    assert (v["verdict"], v["exit_code"]) == ("go", 0) and v["checkpoint_sha256"] == data["sweep_raw"]["checkpoint_sha256"]
    assert "policy" not in json.dumps(v["checks"]).lower().replace("policy not ready", "")
    rep = copy.deepcopy(data["check2_report"])
    rep["metrics"]["action_jitter_high_freq_power_share"] = 1.0
    v = _build(data, check2_report=rep)
    assert (v["verdict"], v["exit_code"]) == ("no-go", 1)
    assert verdict.signature(v)["risks"] == ["check 2: action_jitter_high_freq_power_share"]


def test_verdict_precedence(data):
    rep = copy.deepcopy(data["check2_report"])
    rep["metrics"]["action_jitter_high_freq_power_share"] = 1.0
    assert _build(data, check2_report=rep, newton_raw=None)["exit_code"] == 1      # a flag beats a missing check
    assert _build(data, newton_raw=None)["exit_code"] == 2
    bad = copy.deepcopy(data["sweep_raw"])
    bad["profile"] = "other"
    v = _build(data, sweep_raw=bad)
    assert (v["verdict"], v["exit_code"]) == ("invalid", 3) and v["checks"]["1"]["status"] == "invalid"
    other = pb._v4(2)
    v = _build(data, newton_raw=other["newton_raw"], physx_raw=other["physx_raw"])
    assert v["exit_code"] == 2 and "different checkpoints" in v["problems"][0]


# ---- acceptance ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run():
    inputs, key, made = pb.standin_set()
    res, verdicts = pb.validate(inputs, key)
    return inputs, key, verdicts, res


def test_standin_set_passes_with_the_real_checks(run):
    inputs, key, verdicts, res = run
    assert res["passed"], res
    assert len(key) == 23 and len(set(key)) == 23
    assert all(not k.startswith("P") for k in key)                 # blinded IDs carry no policy name


def _ids(key, pol):
    return sorted((k["seed"], pid) for pid, k in key.items() if k["policy"] == pol)


def test_two_of_three_seeds_is_the_line(run):
    inputs, key, verdicts, _ = run
    vs = copy.deepcopy(verdicts)
    _, pid = _ids(key, "P3")[0]
    vs[pid]["checks"]["2"]["flagged"] = False
    assert acceptance.evaluate(vs, key, None)["criteria"]["A1"]["passed"]       # 2 of 3 still passes
    _, pid = _ids(key, "P3")[1]
    vs[pid]["checks"]["2"]["flagged"] = False
    assert not acceptance.evaluate(vs, key, None)["criteria"]["A1"]["passed"]


def test_p1_needs_friction_or_mass_and_p2_needs_latency(run):
    inputs, key, verdicts, _ = run
    vs = copy.deepcopy(verdicts)
    for _, pid in _ids(key, "P2"):
        vs[pid]["checks"]["1"]["flagged_params"] = ["ground_friction"]          # the wrong parameter
    assert not acceptance.evaluate(vs, key, None)["criteria"]["A2"]["passed"]
    vs = copy.deepcopy(verdicts)
    for _, pid in _ids(key, "P1"):
        vs[pid]["checks"]["1"]["flagged_params"] = ["added_base_mass_kg"]
    assert acceptance.evaluate(vs, key, None)["criteria"]["A2"]["passed"]


def test_invalid_checks_flag_nothing(run):
    inputs, key, verdicts, _ = run
    vs = copy.deepcopy(verdicts)
    for _, pid in _ids(key, "P5"):
        vs[pid]["checks"]["5"]["status"] = "invalid"
    assert not acceptance.evaluate(vs, key, None)["criteria"]["A3"]["passed"]


def test_false_alarms_one_allowed_two_fail(run):
    inputs, key, verdicts, _ = run
    vs = copy.deepcopy(verdicts)
    ids = [x for x in _ids(key, "P0") if x[0] >= 4]
    for s, pid in _ids(key, "P0")[:3]:
        vs[pid]["verdict"] = "no-go"                               # calibration seeds are not held out
    assert acceptance.evaluate(vs, key, None)["criteria"]["A4"]["passed"]
    vs[ids[0][1]]["verdict"] = "no-go"
    assert acceptance.evaluate(vs, key, None)["criteria"]["A4"]["passed"]
    vs[ids[1][1]]["verdict"] = "no-go"
    assert not acceptance.evaluate(vs, key, None)["criteria"]["A4"]["passed"]


def test_harness_validity(run):
    inputs, key, verdicts, _ = run
    vs = copy.deepcopy(verdicts)
    _, pid = _ids(key, "P0")[0]
    vs[pid]["checks"]["5"]["parity"] = "failed"                     # parity applies to every policy, P0 included
    assert not acceptance.evaluate(vs, key, None)["criteria"]["A5"]["passed"]
    vs = copy.deepcopy(verdicts)
    _, pid = _ids(key, "P4")[0]
    vs[pid]["checks"]["5"]["physx_success_rate"] = 0.75             # a planted policy below nominal success
    assert not acceptance.evaluate(vs, key, None)["criteria"]["A5"]["passed"]


def test_replays(run):
    inputs, key, verdicts, _ = run
    pick = acceptance.pick_replays(key, pb.REPLAY_SEED)
    assert pick == acceptance.pick_replays(list(reversed(list(key))), pb.REPLAY_SEED) and len(pick) == 3
    same = {p: copy.deepcopy(verdicts[p]) for p in pick}
    assert acceptance.evaluate(verdicts, key, same)["criteria"]["A6"]["passed"]
    assert not acceptance.evaluate(verdicts, key, dict(list(same.items())[:2]))["criteria"]["A6"]["passed"]
    same[pick[0]]["verdict"], same[pick[0]]["exit_code"] = "invalid", 3
    assert not acceptance.evaluate(verdicts, key, same)["criteria"]["A6"]["passed"]


def test_incomplete_runs_and_bad_keys_fail(run):
    inputs, key, verdicts, _ = run
    vs = dict(verdicts)
    vs.pop(next(iter(vs)))
    assert any("no verdict" in p for p in acceptance.evaluate(vs, key, None)["problems"])
    k2 = copy.deepcopy(key)
    k2[next(iter(k2))]["seed"] = 9
    r = acceptance.evaluate(verdicts, k2, None)
    assert not r["passed"] and any("the plan needs" in p for p in r["problems"])


def test_acceptance_cli(run, tmp_path):
    inputs, key, verdicts, _ = run
    (tmp_path / "v").mkdir()
    (tmp_path / "r").mkdir()
    for pid, v in verdicts.items():
        (tmp_path / "v" / f"{pid}.json").write_text(json.dumps(v))
    for pid in acceptance.pick_replays(key, pb.REPLAY_SEED):
        (tmp_path / "r" / f"{pid}.json").write_text(json.dumps(verdicts[pid]))
    (tmp_path / "key.json").write_text(json.dumps({"policies": key}))
    args = ["--verdicts", str(tmp_path / "v"), "--key", str(tmp_path / "key.json")]
    assert acceptance.main(args + ["--replays", str(tmp_path / "r")]) == 0
    assert acceptance.main(args) == 1                                # A6 needs its 3 replays


# ---- planted bugs ----------------------------------------------------------------------------------------------

def test_planted_bug_test_catches_every_bug_and_leaves_the_checks_alone():
    out = pb.run()
    assert out["control"]["passed"] and out["passed"]
    assert {b["check"] for b in out["bugs"]} == {"1", "2", "5"}
    for b in out["bugs"]:
        assert b["caught"] and not b["validation_passed"], b
        assert b["failed_criteria"] == b["must_fail"], b          # each bug breaks only what its direction predicts
    hides = [b for b in out["bugs"] if b["bug"].startswith("hides flags")]
    alarms = [b for b in out["bugs"] if b["bug"].startswith("false alarms")]
    assert sorted(b["check"] for b in hides) == sorted(b["check"] for b in alarms) == ["1", "2", "5"]
    assert all(b["must_fail"] == ["A4"] for b in alarms) and all("A4" not in b["must_fail"] for b in hides)
    assert robustness.CLIFF_SUCCESS == 0.8 and transfer.NEWTON_FALL_FREE_MIN == 0.8
    assert transfer.ERROR_RATIO_MAX == 2.0 and "v > t\n" in open(es.__file__).read()


def test_a_bug_that_changes_nothing_is_not_caught():
    """The test must be able to say NOT CAUGHT: a harmless edit leaves the validation passing."""
    mod = pb.scratch_copy(transfer, "ERROR_RATIO_MAX = 2.0\n", "ERROR_RATIO_MAX = 2.5\n", tag="harmless")
    inputs, key, _ = pb.standin_set()
    res, _ = pb.validate(inputs, key, {"5": mod.score})
    assert res["passed"]


def test_scratch_copy_refuses_an_ambiguous_or_absent_line():
    with pytest.raises(ValueError, match="0 times"):
        pb.scratch_copy(transfer, "no such line", "x")
    with pytest.raises(ValueError, match="times, not once"):
        pb.scratch_copy(robustness, "return", "x")
