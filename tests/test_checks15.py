"""sim2gate.mvp.checks15 end to end without Isaac Lab: each Isaac Sim launch is replaced by the condition recorder
playing test_conditions' fake robot against the checkpoint it was given."""
import json

import pytest

from sim2gate.checks import isaaclab_conditions as ic
from sim2gate.mvp import checks15
import test_conditions as tc


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd_path = tmp_path / "cmds.json"
    cmd_path.write_text(json.dumps(tc.CMDS))
    seeds = {}
    for s in (1, 2, 3):
        ck = tmp_path / "runs" / f"s{s}" / "model_999.pt"
        ck.parent.mkdir(parents=True)
        ck.write_bytes(f"P0 seed {s}".encode())
        from sim2gate.mvp.baselines import sha256
        seeds[str(s)] = {"checkpoint": {"path": str(ck), "sha256": sha256(str(ck))}}
    (tmp_path / "P0").mkdir()
    (tmp_path / "P0" / "manifest.json").write_text(json.dumps({"seeds": seeds}))
    calls = {"cmds": [], "crash_rc": 0}
    monkeypatch.setattr(ic, "_versions", lambda: {"newton": "1.0", "isaaclab": "3.0"})

    def fake_run(cmd, log):
        open(log, "w").write("fake\n")
        calls["cmds"].append(cmd)
        mode, out = cmd[cmd.index("--mode") + 1], cmd[cmd.index("--out") + 1]
        rest = cmd[cmd.index("--") + 1:]
        ck = rest[rest.index("--checkpoint") + 1]
        newton = "physics=newton_mjwarp" in rest
        rec = ic.ConditionRecorder(mode, out, tc.CMDS, str(cmd_path), rest, episodes=tc.EPS, api=tc.API,
                                   motor_cls=tc.Motor)
        cls = type("E", (tc.CondEnv,), {})
        gym = tc.FakeGym(cls)
        undo = ic.install(rec, wrapper_cls=tc.Wrapper, runner_cls=tc.te.FakeRunner, gym_module=gym)
        try:
            tc.te.FakeRunner().load(ck)
            cfg = tc._cfg()
            cfg.scene.num_envs = rec.num_envs
            env = gym.make("T", cfg=cfg)
            env.sim.physics_manager.__name__ = "NewtonMJWarpManager" if newton else "PhysxManager"
            w = tc.Wrapper(env)
            with pytest.raises(KeyboardInterrupt):
                for _ in range(tc.CondEnv.T + 10):
                    w.step(None)
        finally:
            undo()
        return calls["crash_rc"], ""

    monkeypatch.setattr(checks15, "run", fake_run)
    args = ["--baselines", "P0", "--seeds", "1", "2", "3", "--out", "P0_c15", "--commands", str(cmd_path)]
    return tmp_path, args, calls


def test_runs_calibrates_and_scores(world):
    tmp, args, calls = world
    assert checks15.main(args + ["--calibrate"]) == 0
    assert len(calls["cmds"]) == 9
    for c in calls["cmds"]:
        assert "env.sim.use_newton_actuators=false" in c and "--external_callback" in c
        assert all(o in c for o in ("env.scene.robot.joint_ordering=physx", "env.scene.robot.body_ordering=physx"))
    succ = json.load(open(tmp / "P0_c15" / "success.json"))
    assert succ["threshold_mps"] == pytest.approx(0.075) and len(succ["sources"]) == 3
    for s in (1, 2, 3):
        c1 = json.load(open(tmp / "P0_c15" / f"check1_s{s}.json"))
        c5 = json.load(open(tmp / "P0_c15" / f"check5_s{s}.json"))
        assert c1["status"] == "scored" and c1["params"]["action_latency_ms"]["flagged"]
        assert c5["status"] == "scored" and c5["parity"] == "passed" and c5["flagged"] is False
    m = json.load(open(tmp / "P0_c15" / "manifest.json"))
    assert m["success"]["threshold_mps"] == pytest.approx(0.075)


def test_shutdown_crash_after_output_still_counts(world):
    tmp, args, calls = world
    calls["crash_rc"] = 1
    assert checks15.main(args + ["--calibrate"]) == 0
    m = json.load(open(tmp / "P0_c15" / "manifest.json"))
    assert m["seeds"]["1"]["sweep"]["returncode"] == 1 and "error" not in m["seeds"]["1"]["sweep"]


def test_success_reference_is_never_overwritten_and_only_from_p0_1_to_3(world):
    tmp, args, calls = world
    assert checks15.main(args + ["--calibrate"]) == 0
    assert checks15.main(args + ["--calibrate"]) == 1
    with pytest.raises(SystemExit):
        checks15.main(["--baselines", "P0", "--seeds", "4", "5", "6", "--out", "x", "--calibrate"])


def test_changed_checkpoint_is_refused(world):
    tmp, args, calls = world
    (tmp / "runs" / "s2" / "model_999.pt").write_bytes(b"swapped")
    assert checks15.main(args + ["--calibrate"]) == 1
    assert not (tmp / "P0_c15" / "success.json").exists()
