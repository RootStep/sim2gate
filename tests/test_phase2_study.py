"""v2 phase 2 study driver: training plan, run-failure retry, export, sealing and blind output. No Isaac Lab."""
import json
import os
import shutil
import subprocess

import pytest

from sim2gate.v2 import phase1, phase2_study as st, phase2_score as sc
from sim2gate.mvp import testset
from test_phase2_export import make_robot, BASE_W

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")


@pytest.fixture
def keys(tmp_path):
    priv, pub = tmp_path / "priv.pem", tmp_path / "pub.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(priv)],
                   check=True, capture_output=True)
    subprocess.run(["openssl", "pkey", "-in", str(priv), "-pubout", "-out", str(pub)], check=True, capture_output=True)
    return str(priv), str(pub)


def test_plan_is_the_frozen_seed_table():
    plan = st.study_plan()
    assert [p[2] for p in plan] == list(range(301, 316))
    assert {p[1] for p in plan if p[2] <= 305} == {"R-slip"} and {p[1] for p in plan if p[2] >= 311} == {"Phi-delay"}
    assert plan[0][3] == "Sim2Gate-Velocity-Flat-Go2-P2-R-slip-k0.1"


def fake_world(tmp_path, monkeypatch, fail_first=(), fail_always=()):
    work, p1 = tmp_path / "work", tmp_path / "phase1"
    calls = []

    def fake_tm(work_, name, seed, runner, evidence, measure, require_qualify, task, success, **k):
        calls.append(seed)
        d = os.path.join(work_, "robots", name)
        if os.path.exists(d):
            shutil.rmtree(d)
        if seed in fail_always or (seed in fail_first and calls.count(seed) == 1):
            return {"name": name, "seed": seed, "problems": ["training exited 1; 0 new run folder(s)"],
                    "finished": "t"}
        ev, y = make_robot(tmp_path / "gen", f"{name}_{len(calls)}", BASE_W, behaviour_seed=seed)
        shutil.copytree(ev, os.path.join(d, "evidence"))
        run = tmp_path / "logs" / f"sim2gate_go2_flat_p2_x" / f"run_{seed}_{len(calls)}"
        (run / "params").mkdir(parents=True)
        shutil.copy(y, run / "params" / "env.yaml")
        os.makedirs(os.path.join(d, "measure"), exist_ok=True)
        return {"name": name, "seed": seed, "problems": [], "finished": "t", "run_dir": str(run),
                "checkpoint": {"path": str(run / "model_1999.pt"), "sha256": "c"},
                "summary": {"nominal": {"success_rate": 0.95}, "check2_metrics": {"foot_slip_p95_mps": 0.1 + seed / 1e4}}}
    monkeypatch.setattr(phase1, "train_and_measure", fake_tm)
    # phase 1 controls with records and checkpoints
    robots = {}
    for s in st.CONTROL_SEEDS:
        name = phase1.robot_name(s)
        ev, y = make_robot(tmp_path / "gen", name, BASE_W, behaviour_seed=s)
        shutil.copytree(ev, p1 / "robots" / name / "evidence")
        run = tmp_path / "logs" / "p0v5b" / name
        (run / "params").mkdir(parents=True)
        shutil.copy(y, run / "params" / "env.yaml")
        (run / "model_1999.pt").write_bytes(b"x")
        robots[name] = {"run_dir": str(run), "checkpoint": {"path": str(run / "model_1999.pt"), "sha256": "c"}}
    (p1).mkdir(exist_ok=True)
    (p1 / "state.json").write_text(json.dumps({"robots": robots}))
    monkeypatch.setattr(phase1, "measure_commands", lambda ck, d, task=None: (
        {m: ["true"] for m in st.MEASURE}, {m: os.path.join(d, f"{m}.json") for m in st.MEASURE}))
    monkeypatch.setattr(phase1, "summarize", lambda files, success: {
        "nominal": {"success_rate": 0.96}, "check2_metrics": {"foot_slip_p95_mps": 0.1}})
    monkeypatch.setattr(st, "_success", lambda: {"threshold_mps": 0.1187})
    return work, p1, calls


def test_full_run_exports_20_blind_rows_and_seals_every_labelled_record(tmp_path, monkeypatch, keys, capsys):
    priv, pub = keys
    work, p1, calls = fake_world(tmp_path, monkeypatch, fail_first=(307,))
    blind = tmp_path / "blind"
    assert st.run(str(work), str(p1), pub, str(blind), runner=lambda cmd, log: (0, "")) == 0
    out = capsys.readouterr().out
    assert calls.count(307) == 2 and len(calls) == 16                     # one run failure, retrained once
    for word in ("R-slip", "Phi-slip-mild", "Phi-delay", "s301", "307", "study_s"):
        assert word not in out
    assert "STUDY COMPLETE: 20 robots" in out
    # plain labelled records are gone; only the blind outputs remain
    assert not (work / "key.json").exists() and not (work / "state.json").exists()
    assert not (work / "robots").exists() and not (work / "controls").exists()
    assert not list((tmp_path / "logs" / "sim2gate_go2_flat_p2_x").glob("run_*"))
    t = json.load(open(blind / "features.json"))
    assert len(t["rows"]) == 20 and list(t["rows"]) == sorted(t["rows"])
    # the key holder can open the key, and stage 1 accepts it with the table
    opened = testset.unseal(priv, str(blind / "sealed"), str(tmp_path / "opened"))
    kp = next(p for p in (tmp_path / "opened").rglob("key.json"))
    key = json.load(open(kp))["robots"]
    assert sorted(v["seed"] for v in key.values()) == list(range(101, 106)) + list(range(301, 316))
    assert set(key) == set(t["rows"])
    assert any((tmp_path / "opened").rglob("state.json")) and opened
    pred = sc.stage1(str(blind / "features.json"), str(kp), str(tmp_path / "pred.json"))
    assert len(pred["primary"]) == 20
    rep = json.load(open(blind / "report.json"))
    assert rep["robots"] == 20 and rep["missing_training_records"] == 0


def test_a_bad_public_key_stops_before_any_training(tmp_path, monkeypatch, keys):
    priv, pub = keys
    work, p1, calls = fake_world(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        st.run(str(work), str(p1), priv, str(tmp_path / "blind"), runner=lambda c, l: (0, ""))
    assert calls == []


def test_resumes_without_retraining(tmp_path, monkeypatch, keys):
    work, p1, calls = fake_world(tmp_path, monkeypatch)
    st.train_study(str(work), None)
    n = len(calls)
    st.train_study(str(work), None)
    assert n == 15 and len(calls) == n


def test_a_run_that_fails_twice_is_left_out_of_every_approach(tmp_path, monkeypatch, keys, capsys):
    priv, pub = keys
    work, p1, calls = fake_world(tmp_path, monkeypatch, fail_always=(313,))
    blind = tmp_path / "blind"
    assert st.run(str(work), str(p1), pub, str(blind), runner=lambda cmd, log: (0, "")) == 0
    assert calls.count(313) == 2
    out = capsys.readouterr().out
    assert "STUDY COMPLETE: 19 robots" in out and "run failures excluded 1" in out and "313" not in out
    t = json.load(open(blind / "features.json"))
    assert len(t["rows"]) == 19
    testset.unseal(priv, str(blind / "sealed"), str(tmp_path / "opened"))
    k = json.load(open(next((tmp_path / "opened").rglob("key.json"))))
    assert set(k["robots"]) == set(t["rows"]) and list(k["excluded"].values()) == [
        {"arm": "Phi-delay", "seed": 313, "reason": "run failure twice"}]
    assert 313 not in {v["seed"] for v in sc.load_key(next((tmp_path / "opened").rglob("key.json"))).values()}
