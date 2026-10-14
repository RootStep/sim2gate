"""sim2gate.mvp.baselines end to end without Isaac Lab: a fake trainer writes checkpoints, and check 2 runs for real
on the check 2 tests' fake environment against each checkpoint's actual bytes; calibration then runs on those
reports."""
import json
import os
import shutil

import pytest

from sim2gate.mvp import baselines
from sim2gate.checks.commands import make_command_set
import test_exploits as te


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd_path = tmp_path / "commands.json"
    cmd_path.write_text(json.dumps(make_command_set(te.N, 20261012)))
    calls = {"train": 0, "check2": 0, "fail_train_seed": None, "wrong_checkpoint": False}

    def fake_run(cmd, log):
        open(log, "w").write("fake\n")
        if "sim2gate.checks.isaaclab_exploits" in cmd:
            calls["check2"] += 1
            out, ck = cmd[cmd.index("--out") + 1], cmd[cmd.index("--checkpoint") + 1]
            assert cmd[cmd.index("--commands") + 1] == str(cmd_path) and "physics=physx" in cmd
            data = open(ck, "rb").read() + (b"x" if calls["wrong_checkpoint"] else b"")
            seed = int(ck.split("seed")[-1].split(os.sep)[0])
            env = te.FakeEnv(fall_step=3 + seed)
            (tmp_path / f"c2_{calls['check2']}").mkdir(exist_ok=True)
            rep, path = te._run(env, 60, tmp_path / f"c2_{calls['check2']}", checkpoint=data,
                                command_set=json.load(open(cmd_path)),
                                play_args=cmd[cmd.index("--") + 1:])
            shutil.copy(path, out)
            return 0, ""
        calls["train"] += 1
        seed = int(cmd[cmd.index("--seed") + 1])
        assert "physics=physx" in cmd
        if seed == calls["fail_train_seed"]:
            return 1, ""
        d = tmp_path / "logs" / "rsl_rl" / "unitree_go2_flat" / f"run{calls['train']}_seed{seed}"
        (d / "params").mkdir(parents=True)
        (d / "params" / "env.yaml").write_text(f"seed: {seed}\n")
        (d / "model_149.pt").write_bytes(b"early")
        (d / "model_299.pt").write_bytes(f"P0 seed {seed} weights".encode())
        return 0, ""

    monkeypatch.setattr(baselines, "run", fake_run)
    monkeypatch.setattr(baselines, "train_script", lambda: "/fake/train.py")
    args = ["--policy", "P0", "--seeds", "1", "2", "3", "--commands", str(cmd_path), "--out", "mvp/P0",
            "--num-envs", str(te.N)]
    return tmp_path, args, calls


def test_baselines_train_measure_and_calibrate(world):
    tmp, args, calls = world
    assert baselines.main(args + ["--calibrate"]) == 0
    m = json.load(open(tmp / "mvp/P0/manifest.json"))
    assert calls == {"train": 3, "check2": 3, "fail_train_seed": None, "wrong_checkpoint": False}
    for s in ("1", "2", "3"):
        e = m["seeds"][s]
        assert e["checkpoint"]["path"].endswith("model_299.pt") and len(e["checkpoint"]["sha256"]) == 64
        assert "env.yaml" in e["configs"] and e["check2_report"]["sha256"]
        assert "error" not in e
    assert m["physics"] == "physx" and m["command_set"]["seed"] == 20261012
    th = json.load(open(tmp / "mvp/P0/thresholds.json"))
    assert th["command_set"]["sha256"] == m["command_set"]["sha256"]
    assert th["physics_context"]["overrides"] == ["physics=physx"]
    assert len({s["checkpoint_sha256"] for s in th["sources"]}) == 3


def test_rerun_does_not_retrain_and_never_overwrites_thresholds(world):
    tmp, args, calls = world
    assert baselines.main(args + ["--calibrate"]) == 0
    assert baselines.main(args) == 0 and calls["train"] == 3 and calls["check2"] == 6     # check 2 reruns only
    assert baselines.main(args + ["--calibrate"]) == 1                                    # thresholds exist


def test_failed_training_is_recorded_and_blocks_calibration(world):
    tmp, args, calls = world
    calls["fail_train_seed"] = 2
    assert baselines.main(args + ["--calibrate"]) == 1
    m = json.load(open(tmp / "mvp/P0/manifest.json"))
    assert "exited 1" in m["seeds"]["2"]["error"] and not (tmp / "mvp/P0/thresholds.json").exists()


def test_check2_on_the_wrong_checkpoint_is_an_error(world):
    tmp, args, calls = world
    calls["wrong_checkpoint"] = True
    assert baselines.main(args) == 1
    m = json.load(open(tmp / "mvp/P0/manifest.json"))
    assert all("check 2 measured checkpoint" in m["seeds"][s]["error"] for s in ("1", "2", "3"))


@pytest.mark.parametrize("bad", [["--policy", "P3"], ["--seeds", "4", "5", "6"]])
def test_calibration_only_from_p0_seeds_1_to_3(world, bad):
    tmp, args, calls = world
    a = list(args)
    for i in range(0, len(bad)):
        if bad[i].startswith("--"):
            j = a.index(bad[i]); k = j + 1
            while k < len(a) and not a[k].startswith("--"):
                k += 1
            a[j:k] = [bad[i]] + [x for x in bad[i + 1:] if not x.startswith("--")][:3 if bad[i] == "--seeds" else 1]
    with pytest.raises(SystemExit):
        baselines.main(a + ["--calibrate"])
    assert calls["train"] == 0


def test_command_set_must_match_the_env_count(world):
    tmp, args, calls = world
    with pytest.raises(SystemExit):
        baselines.main([x if x != str(te.N) else str(te.N + 1) for x in args])


def test_the_shipped_command_set_is_the_plans():
    from sim2gate.checks.commands import load_command_set, make_command_set
    doc = load_command_set(baselines.GO2_COMMANDS)
    assert doc["seed"] == 20261012 and len(doc["commands"]) == 64
    assert doc["sha256"] == make_command_set(64, 20261012)["sha256"]          # reproducible from the seed
