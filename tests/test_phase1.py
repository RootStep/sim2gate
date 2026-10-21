"""v2 phase 1 driver (sim2gate.v2.phase1) and the P0 v5 / v5b task definitions. No Isaac Lab."""
import json
import os
import tarfile

import pytest

from sim2gate.v2 import phase1, tasks


def test_p0v5_definition():
    v = tasks.VARIANTS[tasks.P0_TASK_V5]
    assert v["version"] == 5 and v["max_iterations"] == 2000
    assert tasks.P0_V5_PAYLOAD_KG == (0.0, 8.0) and tasks.P0_V5_MOTOR_STRENGTH == (0.8, 1.0)
    assert tasks.P0_V5_STATIC_FRICTION == (0.35, 1.2)
    assert tasks.P0_V5_DYNAMIC_FRICTION[0] == pytest.approx(0.35 * 0.75)
    assert tasks.CALLBACK == ["--external_callback", "sim2gate.v2.tasks.register"]


def test_p0v5b_definition():
    from sim2gate.mvp import tasks as mvp
    v = tasks.VARIANTS[tasks.P0_TASK_V5B]
    assert phase1.TASK == tasks.P0_TASK_V5B and phase1.LABEL == "P0 v5b" and phase1.robot_name(3) == "p0v5b_s3"
    assert v["max_iterations"] == 2000 and v["experiment_name"] == "sim2gate_go2_flat_p0_v5b"
    assert tasks.P0_V5B_PAYLOAD_RAMP_ROUNDS == mvp.P0_V4_DELAY_RAMP_ROUNDS          # the same ramp as the delay
    pay = [c for c in v["changes"] if "payload" in c["what"]]
    assert len(pay) == 1 and pay[0]["mode"] == "reset" and pay[0]["ramp_env_steps"] == [12000, 19200]
    assert any("physics_material" in c["what"] for c in v["changes"]) and any("strength" in c["what"] for c in v["changes"])
    assert tasks.VARIANTS[tasks.P0_TASK_V5]["status"].startswith("superseded")


def test_payload_ramp():
    up = lambda step: tasks.payload_upper_kg(step, 8.0, (12000, 19200))  # noqa: E731
    assert up(0) == 0.0 and up(11999) == 0.0 and up(12000) == 0.0
    assert up(15600) == pytest.approx(4.0) and up(19200) == 8.0 and up(10 ** 6) == 8.0
    assert tasks.payload_upper_kg(0, 8.0, (0, 0)) == 8.0                  # play mode: the full range at once


def test_ramped_payload_term():
    """RampedPayload hands Isaac Lab's own mass term (0, ramped upper bound), with "add", at every reset."""
    calls = []

    class Inner:
        def __init__(self, cfg, env):
            self.cfg = cfg

        def __call__(self, env, env_ids, asset_cfg, params, operation, distribution):
            calls.append((env.common_step_counter, list(env_ids), params, operation, distribution))

    class Base:
        inner_cls = Inner

        def __init__(self, cfg, env):
            self.cfg = cfg

    class TermCfg:
        def __init__(self, func, mode, params):
            self.func, self.mode, self.params = func, mode, params

    class Env:
        common_step_counter = 0

    cls = tasks.ramped_payload_class(Base, TermCfg)
    assert cls.__qualname__ == "RampedPayload" and cls.__module__ == "sim2gate.v2.tasks"
    env = Env()
    params = {"asset_cfg": "base", "mass_distribution_params": (0.0, 8.0), "ramp_env_steps": (12000, 19200),
              "operation": "add", "distribution": "uniform"}
    term = cls(TermCfg(cls, "reset", params), env)
    assert term._inner.cfg.params["operation"] == "add" and term._inner.cfg.params["asset_cfg"] == "base"
    for step in (0, 15600, 30000):
        env.common_step_counter = step
        term(env, [3, 5], **params)
    assert [c[2] for c in calls] == [(0.0, 0.0), (0.0, 4.0), (0.0, 8.0)]
    assert all(c[1] == [3, 5] and c[3:] == ("add", "uniform") for c in calls)
    assert term.last_upper_kg == 8.0


def test_commands():
    t = phase1.train_command(101, "w", "p0v5b_s101", evidence=True)
    i = t.index("--")
    assert "--evidence" in t[:i] and t[t.index("--eta") + 1] == "0"
    assert t[t.index("--task") + 1] == tasks.P0_TASK_V5B and t[t.index("--seed") + 1] == "101"
    assert "sim2gate.v2.tasks.register" in t
    assert "--evidence" not in phase1.train_command(1, "w", "p0v5b_s1", evidence=False)
    cmds, files = phase1.measure_commands("/c/model_1999.pt", "o")
    assert set(cmds) == set(phase1.MEASUREMENTS)
    for m in ("nominal", "sweep", "newton"):
        c = cmds[m]
        assert c[c.index("--envelope") + 1] == "v2"
        assert c[c.index("--task") + 1] == tasks.P0_TASK_V5B and "/c/model_1999.pt" in c
    assert cmds["nominal"][cmds["nominal"].index("--repeat") + 1] == "4"
    assert cmds["newton"][cmds["newton"].index("--repeat") + 1] == "4"
    assert cmds["sweep"][cmds["sweep"].index("--mode") + 1] == "sweep"
    assert "physics=newton_mjwarp" in cmds["newton"] and "physics=isaacsim_physx" in cmds["nominal"]


def _robot(rate, flagged=(), status="scored", problems=()):
    return {"finished": "t", "problems": list(problems),
            "summary": {"nominal": {"success_rate": rate, "episodes": 256, "falls": 0,
                                    "qualifies": rate >= phase1.QUALIFY_MIN},
                        "check1": {"status": status, "flagged": bool(flagged), "reasons": [],
                                   "cliffs": {}, "flagged_params": list(flagged)}}}


def test_pilot_rule():
    ok = {"p0v5b_s1": _robot(0.95), "p0v5b_s2": _robot(0.90)}
    assert phase1.pilot_verdict(ok)["pass"]
    v = phase1.pilot_verdict({"p0v5b_s1": _robot(0.95), "p0v5b_s2": _robot(0.84)})
    assert not v["pass"] and "0.840 < 0.85" in v["reasons"][0]
    v = phase1.pilot_verdict({"p0v5b_s1": _robot(0.95, flagged=["motor_strength_scale"]), "p0v5b_s2": _robot(0.9)})
    assert not v["pass"] and "motor_strength_scale" in v["reasons"][0]
    assert not phase1.pilot_verdict({"p0v5b_s1": _robot(0.95)})["pass"]          # both seeds needed
    assert not phase1.pilot_verdict({"p0v5b_s1": _robot(0.95, status="invalid"), "p0v5b_s2": _robot(0.9)})["pass"]


def test_clean_replaces_robots_that_do_not_qualify(tmp_path, monkeypatch):
    calls = []

    def fake(work, name, seed, runner, evidence, **kw):
        calls.append(seed)
        assert evidence
        d = os.path.join(work, "robots", name)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "train.log"), "w").write("x")
        return _robot(0.7 if seed in (103, 110) else 0.95)
    monkeypatch.setattr(phase1, "train_and_measure", fake)
    work = str(tmp_path / "clean")
    assert phase1.clean(work, None) == 0
    st = json.load(open(os.path.join(work, "state.json")))
    assert len(st["calibration_set"]) == 20 and "p0v5b_s103" not in st["calibration_set"]
    assert calls[-2:] == [121, 122]
    n = len(calls)
    assert phase1.clean(work, None) == 0 and len(calls) == n                  # resumable: nothing retrained
    out = str(tmp_path / "c.tgz")
    assert phase1.bundle(work, out) == 0
    with tarfile.open(out) as t:
        assert "clean/MANIFEST.json" in t.getnames()


def test_clean_stops_when_spares_run_out(tmp_path, monkeypatch):
    monkeypatch.setattr(phase1, "train_and_measure", lambda *a, **k: _robot(0.5))
    assert phase1.clean(str(tmp_path / "c"), None) == 1


def test_clean_refuses_without_a_passed_pilot(tmp_path):
    pw = tmp_path / "pilot"
    pw.mkdir()
    (pw / "state.json").write_text(json.dumps({"robots": {}, "pilot_verdict": {"pass": False}}))
    with pytest.raises(SystemExit):
        phase1.clean(str(tmp_path / "c"), None, pilot_work=str(pw))


def test_smoke_verdict(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    raw = {"envelope": "v2", "profile_record": {"removed_events": ["add_base_mass", "add_payload"]},
           "training_ranges": {"added_base_mass_kg": {"stock_scale": {}, "payload": {
               "params": [0.0, 8.0], "mode": "reset", "ramp_env_steps": [12000, 19200]}},
                               "motor_strength_scale": {"base_legs": [0.8, 1.0]}}}
    (d / "sweep.json").write_text(json.dumps(raw))
    rec = {"problems": [], "_measure_dir": str(d),
           "summary": {"nominal": {}, "check1": {}, "check2_metrics": {}, "newton": {}}}
    assert phase1.smoke_verdict(rec)["pass"]
    raw["training_ranges"]["added_base_mass_kg"]["payload"]["mode"] = "startup"     # P0 v5's payload: rejected now
    (d / "sweep.json").write_text(json.dumps(raw))
    assert not phase1.smoke_verdict(rec)["pass"]
    raw["training_ranges"]["added_base_mass_kg"]["payload"]["mode"] = "reset"
    raw["training_ranges"]["motor_strength_scale"] = "not randomized"
    (d / "sweep.json").write_text(json.dumps(raw))
    assert not phase1.smoke_verdict(rec)["pass"]
