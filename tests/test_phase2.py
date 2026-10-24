"""v2 phase 2, part A: arms, configs, symptoms and the pilot driver. No Isaac Lab."""
import json
import types

import pytest

from sim2gate.v2 import phase2, phase2_arms as arms, tasks
from sim2gate.v2.layer1_limits import load_limit


def fake_cfg():
    w = {"track_lin_vel_xy_exp": 1.5, "feet_slide": -0.25, "action_rate_l2": -0.01, "feet_air_time": 0.25,
         "dof_pos_limits": 0.0}
    rewards = types.SimpleNamespace(**{k: types.SimpleNamespace(weight=v) for k, v in w.items()})
    pm = {"static_friction_range": (0.35, 1.2), "dynamic_friction_range": (0.2625, 0.9), "make_consistent": True}
    legs = types.SimpleNamespace(min_delay=0, max_delay=6)
    return types.SimpleNamespace(rewards=rewards,
                                 events=types.SimpleNamespace(physics_material=types.SimpleNamespace(params=pm)),
                                 scene=types.SimpleNamespace(robot=types.SimpleNamespace(actuators={"base_legs": legs})))


def test_every_arm_keeps_the_same_reward_terms_and_never_zeroes_one():
    base = sorted(vars(fake_cfg().rewards))
    for task, (arm, cid) in arms.all_tasks().items():
        cfg = arms.apply(fake_cfg(), arm, cid)
        assert sorted(vars(cfg.rewards)) == base
        changed = [k for k in base if getattr(cfg.rewards, k).weight != getattr(fake_cfg().rewards, k).weight]
        assert all(getattr(cfg.rewards, k).weight != 0 for k in changed)
    c = arms.apply(fake_cfg(), "R-slip", "k0.1")
    assert c.rewards.feet_slide.weight == pytest.approx(-0.025) and c.rewards.action_rate_l2.weight == -0.01
    c = arms.apply(fake_cfg(), "R-hop", "m10")
    assert c.rewards.feet_air_time.weight == pytest.approx(2.5)
    c = arms.apply(fake_cfg(), "Phi-slip", "low")
    assert c.events.physics_material.params["dynamic_friction_range"] == pytest.approx((0.15, 0.375))
    c = arms.apply(fake_cfg(), "Phi-jitter", "nodelay")
    assert c.scene.robot.actuators["base_legs"].max_delay == 0


def test_refuses_to_zero_a_term(monkeypatch):
    monkeypatch.setitem(arms.ARMS["R-slip"]["candidates"][0], "reward_scale", {"feet_slide": 0.0})
    with pytest.raises(ValueError):
        arms.apply(fake_cfg(), "R-slip", "k0.1")


def test_tasks_registered_for_every_candidate():
    v = tasks._p2_variants()
    assert set(v) == set(arms.all_tasks()) and len(v) == 11
    assert len({x["env_class"] for x in v.values()}) == 11 and len({x["experiment_name"] for x in v.values()}) == 11
    assert all(x["base_task"] == arms.BASE_TASK and x["max_iterations"] == 2000 for x in v.values())


def test_seed_table_is_fixed():
    t = phase2.seed_table()
    assert t[("R-slip", "k0.1")] == (201, 202) and t[("R-hop", "m30")] == (217, 218)
    assert len({s for p in t.values() for s in p}) == 18


def test_check_config_reads_isaac_lab_yaml(tmp_path):
    y = tmp_path / "env.yaml"
    y.write_text("rewards:\n  feet_slide: {weight: -0.025}\n  feet_air_time: {weight: 0.25}\n  undesired_contacts: null\n"
                 "events:\n  physics_material:\n    params:\n      static_friction_range: !!python/tuple\n"
                 "      - 0.2\n      - 0.5\n")
    bad, terms = phase2.check_config("R-slip", "k0.1", str(y))
    assert not bad and terms == ["feet_air_time", "feet_slide"]
    assert phase2.check_config("Phi-slip", "low", str(y))[0] == []
    assert phase2.check_config("R-hop", "m10", str(y))[0]                     # weight not scaled: a problem


def test_symptoms_use_the_frozen_limit_and_hop_margins():
    lim = load_limit()
    slip_t = lim["limit"] * lim["clean_maxima"]["foot_slip_p95_mps"]
    assert slip_t == pytest.approx(0.1185, abs=1e-4)
    assert arms.symptom_present("slip", {"foot_slip_p95_mps": slip_t * 1.001}, {})
    assert not arms.symptom_present("slip", {"foot_slip_p95_mps": lim["clean_maxima"]["foot_slip_p95_mps"] * 1.03}, {})
    jit_t = lim["limit"] * lim["clean_maxima"]["action_jitter_high_freq_power_share"]
    assert jit_t == pytest.approx(0.00986, abs=1e-5)
    assert arms.symptom_present("jitter", {"action_jitter_high_freq_power_share": 0.0099}, {})
    ref = {"flight_share_low_speed": 0.004, "base_vz_rms_low_speed_mps": 0.05}
    th = arms.symptom_thresholds(ref)["hop"]
    assert th == {"flight_share_low_speed": 0.02, "base_vz_rms_low_speed_mps": pytest.approx(0.075)}   # floor applies
    assert arms.symptom_present("hop", {"flight_share_low_speed": 0.02, "base_vz_rms_low_speed_mps": 0.08}, ref)
    assert not arms.symptom_present("hop", {"flight_share_low_speed": 0.3, "base_vz_rms_low_speed_mps": 0.07}, ref)
    assert arms.symptom_thresholds({"flight_share_low_speed": 0.1, "base_vz_rms_low_speed_mps": 0.1})["hop"][
        "flight_share_low_speed"] == pytest.approx(0.15)


def test_pilot_tries_candidates_in_order_and_drops_an_arm(tmp_path, monkeypatch):
    calls = []
    ctl = {f"p0v5b_s{s}": {"check2_metrics": {"flight_share_low_speed": 0.01, "base_vz_rms_low_speed_mps": 0.05}}
           for s in phase2.CONTROL_SEEDS}
    monkeypatch.setattr(phase2, "remeasure_controls", lambda *a, **k: ctl)
    monkeypatch.setattr(phase2, "_env_yaml", lambda rec: "env.yaml")
    monkeypatch.setattr(phase2, "check_config", lambda *a: ([], ["feet_slide"]))

    def fake_tm(work, name, seed, runner, evidence, measure, require_qualify, task, success, **k):
        calls.append((task, seed))
        arm, cid = arms.all_tasks()[task]
        sym = not (arm == "R-slip" and cid == "k0.1") and arm != "Phi-jitter"
        met = {"foot_slip_p95_mps": 0.5 if sym else 0.05, "action_jitter_high_freq_power_share": 0.05 if sym else 0.001,
               "flight_share_low_speed": 0.3 if sym else 0.0, "base_vz_rms_low_speed_mps": 0.4 if sym else 0.0}
        return {"finished": "t", "problems": [], "run_dir": None,
                "summary": {"nominal": {"qualifies": True, "success_rate": 0.95}, "check2_metrics": met}}
    monkeypatch.setattr(phase2.phase1, "train_and_measure", fake_tm)
    work = str(tmp_path / "p")
    assert phase2.pilot(work, "unused", None) == 0
    st = json.load(open(tmp_path / "p" / "state.json"))
    assert st["result"] == {"R-slip": "k0.03", "Phi-slip": "low", "R-jitter": "k0.1", "Phi-jitter": None,
                            "R-hop": "m10"}
    assert ("Sim2Gate-Velocity-Flat-Go2-P2-R-slip-k0.03", 203) in calls
    assert ("Sim2Gate-Velocity-Flat-Go2-P2-Phi-slip-high", 207) not in calls        # first candidate passed
    assert st["arms"]["Phi-jitter"]["dropped"] and len(calls) == 12
    n = len(calls)
    assert phase2.pilot(work, "unused", None) == 0 and len(calls) == n               # resumable: nothing retrained


def test_amendment1_arms_and_delay_symptom():
    c = arms.apply(fake_cfg(), "Phi-slip-mild", "mild")
    assert c.events.physics_material.params["static_friction_range"] == (0.3, 0.6)
    assert c.events.physics_material.params["dynamic_friction_range"] == pytest.approx((0.225, 0.45))
    assert arms.apply(fake_cfg(), "Phi-delay", "nodelay").scene.robot.actuators["base_legs"].max_delay == 0
    assert set(arms.AMENDMENT1_SEEDS.values()) == {(219, 220), (221, 222)}
    assert not set(s for p in arms.AMENDMENT1_SEEDS.values() for s in p) & \
        set(s for p in phase2.seed_table().values() for s in p)                  # new seeds only
    assert arms.delay_symptom({"status": "scored", "flagged_params": ["action_latency_ms"]})
    assert not arms.delay_symptom({"status": "scored", "flagged_params": ["ground_friction"]})
    assert not arms.delay_symptom({"status": "invalid", "flagged_params": ["action_latency_ms"]})


def test_pilot2_hard_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(phase2, "_env_yaml", lambda rec: "env.yaml")
    monkeypatch.setattr(phase2, "check_config", lambda *a: ([], ["feet_slide"]))

    def fake_tm(work, name, seed, runner, evidence, measure, require_qualify, task, success, **k):
        assert "sweep" in measure
        arm, _ = arms.all_tasks()[task]
        return {"finished": "t", "problems": [], "run_dir": None,
                "summary": {"nominal": {"qualifies": True, "success_rate": 0.95},
                            "check2_metrics": {"foot_slip_p95_mps": 0.05},                  # mild friction: no slip
                            "check1": {"status": "scored", "flagged_params": ["action_latency_ms"],
                                       "cliffs": {"action_latency_ms": 25.0}}}}
    monkeypatch.setattr(phase2.phase1, "train_and_measure", fake_tm)
    assert phase2.pilot2(str(tmp_path / "p"), None) == 0
    st = json.load(open(tmp_path / "p" / "state.json"))
    assert st["result"] == {"Phi-slip-mild": False, "Phi-delay": True}
    assert st["cause_study"].startswith("not tested")


def test_smoke2_checks_each_amendment_arm(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(phase2, "_env_yaml", lambda rec: "env.yaml")
    monkeypatch.setattr(phase2, "check_config", lambda arm, cid, y: (seen.append(arm) or [], ["feet_slide"]))
    monkeypatch.setattr(phase2.phase1, "train_and_measure",
                        lambda *a, **k: (k["iterations"] == 2 and not k["measure"]) and
                        {"finished": "t", "problems": [], "run_dir": None, "summary": {}})
    assert phase2.smoke2(str(tmp_path / "s"), None) == 0 and seen == ["Phi-slip-mild", "Phi-delay"]
    monkeypatch.setattr(phase2, "check_config", lambda *a: (["static friction wrong"], []))
    assert phase2.smoke2(str(tmp_path / "s2"), None) == 1
