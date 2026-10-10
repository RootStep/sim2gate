"""smoke_go2 end to end with a stand-in for Isaac Lab's train.py (toy Mountain Car through real rsl-rl-lib)."""
import json
import os
import sys
import textwrap
import pytest

pytest.importorskip("rsl_rl")
from sim2gate.training import smoke_go2  # noqa: E402

FAKE_TRAIN = textwrap.dedent('''
    import argparse, contextlib, io, sys, time, torch
    from rsl_rl.runners import OnPolicyRunner
    from mountain_car_env import VecMountainCar
    p = argparse.ArgumentParser(); p.add_argument("--task"); p.add_argument("--seed", type=int)
    p.add_argument("--max_iterations", type=int); p.add_argument("--headless", action="store_true")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    env = VecMountainCar(num_envs=16, reward="dense", max_episode_length=64, seed=a.seed, init_velocity=True)
    mlp = {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "elu", "obs_normalization": False}
    cfg = {"num_steps_per_env": 48, "save_interval": 10**9, "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
           "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 4},
           "actor": dict(mlp, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0}),
           "critic": dict(mlp)}
    with contextlib.redirect_stdout(io.StringIO()):
        runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    orig = runner.alg.update
    def update(*x, **k):
        t = time.time(); out = orig(*x, **k); print(f"   Iteration time: {0.5 + (time.time() - t):.2f}s"); return out
    runner.alg.update = update
    with contextlib.redirect_stdout(sys.__stdout__):
        runner.learn(a.max_iterations, init_at_random_ep_len=True)
''')


def test_smoke_end_to_end(tmp_path, monkeypatch):
    script = tmp_path / "train.py"; script.write_text(FAKE_TRAIN)
    here = os.path.dirname(__file__)
    monkeypatch.setenv("ISAAC_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([here, os.environ.get("PYTHONPATH", "")]))
    out = tmp_path / "out"
    rc = smoke_go2.main(["--iterations", "3", "--out", str(out), "--isaaclab-script", str(script), "--task", "Fake"])
    report = json.load(open(out / "report.json"))
    failed = [c for c in report["checks"] if not c["ok"]]
    assert rc == 0 and report["result"] == "PASS", failed
    assert len(report["summaries"]) == 2


def test_checks_fail_on_missing_records():
    res = smoke_go2.checks(30, [], [], {"stock": 0, "eta0": 1, "eta1": 0})
    assert not all(r["ok"] for r in res)


def _rec(**sig):
    base = {"num_samples": 10, "identity_residual": 1e-7, "trainer_signal_residual": 1e-7,
            "trainer_normalization_residual": 1e-7, "normalization_group": "rollout", "rho": 0.9,
            "predicted_rho": 0.9, "timeouts": 1, "terminations": 1}
    base.update(sig)
    return {"eta": 1.0, "eta_is_training_eta": True, "signals": base, "reward_terms": {"a": 1.0},
            "reward_term_abs_share": {"a": 1.0}}


def test_checks_reject_malformed_numbers():
    rc = {"stock": 0, "eta0": 0, "eta1": 0}
    good0 = [dict(_rec(), eta_is_training_eta=False)]
    for bad in (float("-inf"), float("nan"), None):
        res = smoke_go2.checks(1, good0, [_rec(identity_residual=bad)], rc)
        assert not next(r for r in res if r["name"].startswith("BSRS identity"))["ok"]
    res = smoke_go2.checks(1, good0, [_rec()], {})
    assert not res[0]["ok"]                      # no return codes at all is a failure, not a pass
    res = smoke_go2.checks(1, good0, [_rec(trainer_signal_residual=0.3)], rc)
    assert not next(r for r in res if r["name"].startswith("reconstruction"))["ok"]


def test_overhead_check():
    base = "\n".join(f"Iteration time: {0.59:.2f}s" for _ in range(10))
    slow = "\n".join(f"Iteration time: {0.70:.2f}s" for _ in range(10))
    assert smoke_go2.overhead_check(base, base)["ok"]
    assert not smoke_go2.overhead_check(base, slow)["ok"]
