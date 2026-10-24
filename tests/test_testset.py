"""Unattended training and blinding of the test set (sim2gate.mvp.testset), without Isaac Sim: training writes a fake
run folder, the walking check a fake nominal run. Sealing uses the real OpenSSL with a throwaway key pair."""
import json
import os
import random
import shutil
import subprocess

import pytest

from sim2gate.mvp import tasks, testset
from sim2gate.mvp.baselines import sha256

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out",
                    "priv.pem"], check=True, capture_output=True)
    subprocess.run(["openssl", "pkey", "-in", "priv.pem", "-pubout", "-out", "pub.pem"], check=True,
                   capture_output=True)
    seeds = {}
    for s in (1, 2, 3):
        ck = tmp_path / "logs" / "rsl_rl" / "sim2gate_go2_flat_p0_v4" / f"cal{s}" / "model_1999.pt"
        ck.parent.mkdir(parents=True)
        ck.write_bytes(f"calibration {s}".encode())
        seeds[str(s)] = {"checkpoint": {"path": str(ck), "sha256": sha256(str(ck))}}
    (tmp_path / "cal.json").write_text(json.dumps({"task": tasks.P0_TASK, "seeds": seeds}))
    monkeypatch.setattr(testset, "train_script", lambda: "train.py")
    fail = {"walk": set()}                  # (task, seed) whose walking check fails
    calls = []
    counter = {"n": 0}

    def fake(cmd, log):
        calls.append(cmd)
        open(log, "w").write(" ".join(cmd))
        task = cmd[cmd.index("--task") + 1]
        if "train.py" in cmd:
            seed = int(cmd[cmd.index("--seed") + 1])
            counter["n"] += 1
            rd = tmp_path / "logs" / "rsl_rl" / tasks.VARIANTS[task]["experiment_name"] / f"run{counter['n']:03d}"
            rd.mkdir(parents=True)
            for it in (0, 50, 1999):
                (rd / f"model_{it}.pt").write_bytes(f"{task} {seed} {it}".encode())
            (rd / "params").mkdir()
            (rd / "params" / "env.yaml").write_text(f"task: {task}\n")
            return 0, "Learning iteration 1999/2000\n"
        ck = cmd[cmd.index("--checkpoint") + 1]
        out = cmd[cmd.index("--out") + 1]
        seed = int(open(ck, "rb").read().decode().split()[1])
        bad = (task, seed) in fail["walk"]
        json.dump({"checkpoint_sha256": sha256(ck), "episodes": [
            {"env": i, "group": 0, "fell": bad and i % 2 == 0, "steps": 1000, "mean_error_mps": 0.06}
            for i in range(64)]}, open(out, "w"))
        return 0, ""

    args = dict(pubkey="pub.pem", calibration="cal.json", work="work", blind="blind", sealed="sealed",
                runner=fake, rng=random.Random(3))
    return tmp_path, args, fail, calls


def _key(tmp_path):
    testset.unseal("priv.pem", "sealed", "unsealed")
    return json.load(open(tmp_path / "unsealed" / "work" / "key.json"))


def test_trains_walks_blinds_and_seals(world, capsys):
    tmp, args, fail, calls = world
    assert testset.train_all(**args) == 0
    trained = [c for c in calls if "train.py" in c]
    assert len(trained) == 20
    out = capsys.readouterr().out
    for word in ("P1", "P3", "P5", "Velocity", "seed"):
        assert word not in out                                   # the console names no policy
    assert not (tmp / "work").exists()                           # labelled records only inside the sealed files
    assert not list((tmp / "logs" / "rsl_rl").glob("sim2gate_go2_flat_p[1-5]_v*/*"))
    assert sorted(os.listdir(tmp / "sealed")) == sorted([testset.SEALED_DATA, testset.SEALED_KEY,
                                                         "sealed_manifest.json"])
    man = json.load(open(tmp / "blind" / "manifest.json"))
    assert len(man["policies"]) == 23
    key = _key(tmp)
    assert {k["policy"] for k in key["policies"].values()} == {"P0", "P1", "P2", "P3", "P4", "P5"}
    assert sorted(k["seed"] for k in key["policies"].values() if k["policy"] == "P0") == [1, 2, 3, 4, 5, 6, 7, 8]
    for pid, k in key["policies"].items():
        assert man["policies"][pid]["sha256"] == k["sha256"]
    runs = list((tmp / "unsealed" / "work" / "runs").glob("*/*"))
    assert len(runs) == 20 and all(sorted(p.name for p in r.glob("model_*.pt")) == ["model_1999.pt"] for r in runs)
    with pytest.raises(testset.OpenSSLError):           # only the private key opens it
        testset.unseal("pub.pem", "sealed", "nope")


def test_a_policy_that_does_not_walk_is_retrained_with_a_new_seed(world):
    tmp, args, fail, calls = world
    fail["walk"].add((tasks.PLANTED_TASKS["P3"], 2))
    assert testset.train_all(**args) == 0
    trained = [(c[c.index("--task") + 1], int(c[c.index("--seed") + 1])) for c in calls if "train.py" in c]
    assert len(trained) == 21 and (tasks.PLANTED_TASKS["P3"], 12) in trained
    state = json.load(open(tmp / "unsealed" / "work" / "state.json")) if testset.unseal(
        "priv.pem", "sealed", "unsealed") else None
    p3 = next(j for j in state["jobs"] if j["policy"] == "P3" and j["base_seed"] == 2)
    assert [a["seed"] for a in p3["attempts"]] == [2, 12] and p3["attempts"][0]["walk"]["success_rate"] == 0.5


def test_p5_moves_to_the_fallback_ground_if_any_seed_fails(world):
    tmp, args, fail, calls = world
    fail["walk"].add((tasks.PLANTED_TASKS["P5"], 1))
    assert testset.train_all(**args) == 0
    key = _key(tmp)
    p5 = [k for k in key["policies"].values() if k["policy"] == "P5"]
    assert len(p5) == 3 and {k["task"] for k in p5} == {tasks.P5_FALLBACK_TASK}
    assert sorted(k["seed"] for k in p5) == [1, 2, 3]


def test_three_failures_stop_for_a_decision(world):
    tmp, args, fail, calls = world
    t = tasks.PLANTED_TASKS["P4"]
    fail["walk"].update({(t, 3), (t, 13), (t, 23)})
    assert testset.train_all(**args) == 1
    assert not (tmp / "blind").exists() and not (tmp / "sealed").exists()
    state = json.load(open(tmp / "work" / "state.json"))
    assert any(j["status"] == "failed" for j in state["jobs"])


def test_check_refuses_a_bad_key_and_missing_calibration(world):
    tmp, args, fail, calls = world
    with pytest.raises(testset.PrivateKeyError):
        testset.check_pubkey("priv.pem")
    (tmp / "junk.pem").write_text("-----BEGIN PUBLIC KEY-----\nnot a key\n-----END PUBLIC KEY-----\n")
    with pytest.raises(testset.OpenSSLError):
        testset.check_pubkey("junk.pem")
    (tmp / "logs" / "rsl_rl" / "sim2gate_go2_flat_p0_v4" / "cal2" / "model_1999.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="seed 2"):
        testset._calibration("cal.json")
    assert testset.main(["check", "--pubkey", "pub.pem", "--calibration", "cal.json"]) == 2


def test_smoke_runs_the_pipeline_once_and_cleans_up(world, capsys):
    tmp, args, fail, calls = world
    assert testset.smoke("pub.pem", "smoke", runner=args["runner"]) == 0
    assert "SMOKE OK" in capsys.readouterr().out
    assert not (tmp / "smoke").exists()
    assert not list((tmp / "logs" / "rsl_rl").glob("sim2gate_go2_flat_p2_v1/*"))
    assert "--max_iterations" in [c for c in calls if "train.py" in c][0]


def test_openssl_runs_without_a_foreign_library_path(world, monkeypatch):
    """Inside Isaac Sim's Python, LD_LIBRARY_PATH points at bundled libraries the system openssl must not load."""
    tmp, args, fail, calls = world
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp / "not-a-real-lib-dir"))
    monkeypatch.setenv("OPENSSL_CONF", str(tmp / "missing.cnf"))     # would make openssl fail if passed through
    assert "Public-Key" in testset.check_pubkey("pub.pem")
    (tmp / "junk.pem").write_text("-----BEGIN PUBLIC KEY-----\nnot a key\n-----END PUBLIC KEY-----\n")
    with pytest.raises(testset.OpenSSLError, match="failed"):
        testset.check_pubkey("junk.pem")


def test_a_stopped_run_resumes_only_after_a_redefinition(world, capsys):
    tmp, args, fail, calls = world
    t1 = tasks.P3_V1_TASK
    fail["walk"].update({(t1, 3), (t1, 13), (t1, 23)})
    import json as _j
    st = None
    # reproduce the Oct 16 situation: the P3 jobs were planned on P3-v1
    orig = testset.plan
    testset.plan = lambda: [(p, s, t1 if p == "P3" else t) for p, s, t in orig()]
    try:
        assert testset.train_all(**args) == 1
    finally:
        testset.plan = orig
    assert testset.train_all(**args) == 1                       # refuses while a job waits for a decision
    assert "waiting for a decision" in capsys.readouterr().out
    n_before = len([c for c in calls if "train.py" in c])
    with pytest.raises(ValueError):
        testset.redefine("work", "P3", tasks.PLANTED_TASKS["P5"], "C9")
    assert testset.redefine("work", "P3", tasks.P3_V2_TASK, "C9") == 3
    assert testset.train_all(**args) == 0
    trained = [(c[c.index("--task") + 1], int(c[c.index("--seed") + 1])) for c in calls if "train.py" in c][n_before:]
    assert sorted(s for t, s in trained if t == tasks.P3_V2_TASK) == [1, 2, 3]
    assert all(t != t1 for t, s in trained)
    key = _key(tmp)
    p3 = [k for k in key["policies"].values() if k["policy"] == "P3"]
    assert {k["task"] for k in p3} == {tasks.P3_V2_TASK} and sorted(k["seed"] for k in p3) == [1, 2, 3]
    st = _j.load(open(tmp / "unsealed" / "work" / "state.json"))
    assert st["redefinitions"][0]["change"] == "C9"


def test_p3_v2_definition():
    v1, v2 = tasks.VARIANTS[tasks.P3_V1_TASK], tasks.VARIANTS[tasks.P3_V2_TASK]
    assert v2["changes"][0]["static_friction_range"] == [0.3, 0.5] and v1["changes"][0]["static_friction_range"] == [0.2, 0.4]
    assert tasks.PLANTED_TASKS["P3"] == tasks.P3_V2_TASK and "superseded" in v1["status"]
    assert v2["experiment_name"] != v1["experiment_name"]
    assert tasks._entry(v1)["env_cfg_entry_point"].endswith("Go2FlatP3EnvCfg")
    assert tasks._entry(v2)["env_cfg_entry_point"].endswith("Go2FlatP3V2EnvCfg")
