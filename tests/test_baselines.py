"""sim2gate.mvp.baselines end to end without Isaac Lab: a fake trainer writes checkpoints, and check 2 runs for real
on the check 2 tests' fake environment against each checkpoint's actual bytes; calibration then runs on those
reports."""
import json
import os
import shutil

import pytest
import torch

from sim2gate.mvp import baselines
from sim2gate.checks.commands import make_command_set
import test_exploits as te


# The tail of P0 v1 seed 3's training log (Oct 14): training finished, then Isaac Sim died while closing
CRASH_LOG = """                       Learning iteration 298/300
...
                       Learning iteration 299/300
...
Thread 0x0000122bc207c780 (most recent call first):
  File "/isaac-sim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py", line 1007 in close
  File "/workspaces/isaaclab_arena/submodules/IsaacLab/source/isaaclab/isaaclab/app/sim_launcher.py", line 594 in launch_simulation
There was an error running python
"""


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd_path = tmp_path / "commands.json"
    cmd_path.write_text(json.dumps(make_command_set(te.N, 20261012)))
    calls = {"train": 0, "check2": 0, "fail_train_seed": None, "wrong_checkpoint": False, "tracking_off": 0.0,
             "cmds": [], "shutdown_crash_seed": None, "crash_text": None}

    def fake_run(cmd, log):
        open(log, "w").write("fake\n")
        if "sim2gate.checks.isaaclab_exploits" in cmd:
            calls["check2"] += 1
            out, ck = cmd[cmd.index("--out") + 1], cmd[cmd.index("--checkpoint") + 1]
            assert cmd[cmd.index("--commands") + 1] == str(cmd_path) and "physics=isaacsim_physx" in cmd
            data = open(ck, "rb").read() + (b"x" if calls["wrong_checkpoint"] else b"")
            seed = int(ck.split("seed")[-1].split(os.sep)[0])
            env = te.FakeEnv(fall_step=3 + seed)
            calls["cmds"].append(cmd)
            cmds = torch.tensor(json.load(open(cmd_path))["commands"], dtype=torch.float32)
            v = torch.zeros(te.N, 3); v[:, :2] = cmds[:, :2] + calls["tracking_off"]   # tracks its command
            env.robot.data.root_lin_vel_b = te._Buf(v)
            (tmp_path / f"c2_{calls['check2']}").mkdir(exist_ok=True)
            rep, path = te._run(env, 60, tmp_path / f"c2_{calls['check2']}", checkpoint=data,
                                command_set=json.load(open(cmd_path)),
                                play_args=cmd[cmd.index("--") + 1:])
            shutil.copy(path, out)
            return 0, ""
        calls["train"] += 1
        calls["cmds"].append(cmd)
        seed = int(cmd[cmd.index("--seed") + 1])
        assert "physics=isaacsim_physx" in cmd
        if seed == calls["fail_train_seed"]:
            return 1, ""
        d = tmp_path / "logs" / "rsl_rl" / "unitree_go2_flat" / f"run{calls['train']}_seed{seed}"
        (d / "params").mkdir(parents=True)
        (d / "params" / "env.yaml").write_text(f"seed: {seed}\n")
        (d / "model_149.pt").write_bytes(b"early")
        (d / "model_299.pt").write_bytes(f"P0 seed {seed} weights".encode())
        if seed == calls["shutdown_crash_seed"]:
            return 1, calls["crash_text"] or CRASH_LOG
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
    assert calls["train"] == 3 and calls["check2"] == 3
    for s in ("1", "2", "3"):
        e = m["seeds"][s]
        assert e["checkpoint"]["path"].endswith("model_299.pt") and len(e["checkpoint"]["sha256"]) == 64
        assert "env.yaml" in e["configs"] and e["check2_report"]["sha256"]
        assert "error" not in e
    assert m["physics"] == "isaacsim_physx" and m["command_set"]["seed"] == 20261012
    th = json.load(open(tmp / "mvp/P0/thresholds.json"))
    assert th["command_set"]["sha256"] == m["command_set"]["sha256"]
    assert th["physics_context"]["overrides"] == ["physics=isaacsim_physx"]
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


def test_a_crash_while_isaac_sim_shuts_down_keeps_the_finished_training(world):
    tmp, args, calls = world
    calls["shutdown_crash_seed"] = 3
    assert baselines.main(args + ["--iterations", "300", "--calibrate"]) == 0
    e = json.load(open(tmp / "mvp/P0/manifest.json"))["seeds"]["3"]
    assert "error" not in e and e["train_returncode"] == 1 and e["checkpoint"]["path"].endswith("model_299.pt")
    assert e["train_shutdown_crash"]["last_iteration"] == "299/300" and calls["check2"] == 3
    assert (tmp / "mvp/P0/thresholds.json").exists()


@pytest.mark.parametrize("why, text, iterations", [
    ("training stopped early", CRASH_LOG.replace("299/300", "250/300"), "300"),
    ("the final checkpoint is not the expected one", CRASH_LOG, "1000"),
    ("the crash was not in SimulationApp.close", CRASH_LOG.replace("in close", "in step"), "300"),
    ("the crash came before the last iteration line", CRASH_LOG.split("Thread")[1] + CRASH_LOG.split("Thread")[0], "300"),
])
def test_other_crashes_stay_training_failures(world, why, text, iterations):
    tmp, args, calls = world
    calls["shutdown_crash_seed"], calls["crash_text"] = 2, text
    assert baselines.main(args + ["--iterations", iterations]) == 1, why
    e = json.load(open(tmp / "mvp/P0/manifest.json"))["seeds"]["2"]
    assert e["error"] == "training exited 1" and "train_shutdown_crash" not in e and "checkpoint" not in e


def test_shutdown_crash_needs_a_known_iteration_count():
    ck = "/x/model_299.pt"
    assert baselines.shutdown_crash(1, CRASH_LOG, ck, 300)["final_checkpoint"] == "model_299.pt"
    assert baselines.shutdown_crash(0, CRASH_LOG, ck, 300) is None          # a clean exit needs no excuse
    assert baselines.shutdown_crash(1, CRASH_LOG, ck, None) is None
    assert baselines.shutdown_crash(1, CRASH_LOG, None, 300) is None


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



def test_p0_uses_the_registered_variant_with_its_callback(world):
    from sim2gate.mvp.tasks import P0_TASK, VARIANTS
    tmp, args, calls = world
    assert baselines.main(args) == 0
    for cmd in calls["cmds"]:
        assert cmd[cmd.index("--task") + 1] == P0_TASK
        assert cmd[cmd.index("--external_callback") + 1] == "sim2gate.mvp.tasks.register"
    m = json.load(open(tmp / "mvp/P0/manifest.json"))
    assert m["task_variant"] == VARIANTS[P0_TASK] and m["baseline_criteria"] == baselines.BASELINE_CRITERIA
    assert all(m["seeds"][s]["acceptance"]["accepted"] for s in ("1", "2", "3"))


def test_baselines_that_miss_the_criteria_are_not_calibrated(world):
    tmp, args, calls = world
    calls["tracking_off"] = 0.3                                   # 0.42 m/s tracking error > 0.15
    assert baselines.main(args + ["--calibrate"]) == 1
    m = json.load(open(tmp / "mvp/P0/manifest.json"))
    acc = m["seeds"]["1"]["acceptance"]
    assert acc["accepted"] is False and acc["details"]["passed"]["tracking"] is False
    assert "baseline criteria" in m["calibration"]["error"] and not (tmp / "mvp/P0/thresholds.json").exists()


def test_task_module_imports_without_isaac_lab_and_versions_the_variant():
    import importlib
    t = importlib.import_module("sim2gate.mvp.tasks")
    v = t.VARIANTS[t.P0_TASK]
    assert t.P0_TASK == t.P0_TASK_V2 and t.P0_TASK.endswith("-v2") and v["version"] == 2
    assert v["max_iterations"] == 1000 and v["changes"][0]["weight"] == -0.25
    assert "not validated" in v["changes"][0]["weight_status"]
    v1 = t.VARIANTS[t.P0_TASK_V1]                                  # v1 stays registered, marked superseded
    assert v1["max_iterations"] == 300 and "superseded" in v1["status"]
    assert [c for c in v["changes"] if "feet_slide" in c["what"]] == [c for c in v1["changes"]]  # same penalty
    for name in ("Go2FlatP0EnvCfgV1", "Go2FlatP0EnvCfgV2", "Go2FlatP0PPORunnerCfgV2"):
        with pytest.raises(ImportError):
            getattr(t, name)                                       # built only where Isaac Lab is installed
