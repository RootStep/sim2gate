"""Update-level identity (sim2gate.training.update_identity) on real rsl-rl-lib, CPU, toy Mountain Car."""
import contextlib
import io
import json

import pytest
import torch

pytest.importorskip("rsl_rl")
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from mountain_car_env import VecMountainCar  # noqa: E402
from test_monitor import cfg  # noqa: E402
from sim2gate.training import update_identity as ui  # noqa: E402
from sim2gate.training.monitor import TrainingMonitor  # noqa: E402

PPO = "rsl_rl.algorithms:PPO"


def _train(out=None, iterations=1, per_mb=False, obs_norm=False, seed=1):
    undo = ui.install(str(out)) if out else (lambda: None)
    try:
        torch.manual_seed(seed)
        env = VecMountainCar(num_envs=16, reward="sparse", max_episode_length=64, seed=seed, init_velocity=True)
        with contextlib.redirect_stdout(io.StringIO()):
            runner = OnPolicyRunner(env, cfg(PPO, 0.0, per_mb, obs_norm), log_dir=None, device="cpu")
            runner.learn(iterations, init_at_random_ep_len=True)
    finally:
        undo()
    a = runner.alg
    return torch.cat([p.detach().reshape(-1) for p in list(a.actor.parameters()) + list(a.critic.parameters())])


@pytest.mark.parametrize("per_mb,obs_norm", [(False, False), (True, False), (False, True)])
def test_monitor_leaves_the_update_identical(tmp_path, per_mb, obs_norm):
    out = tmp_path / "ui.json"
    _train(out, per_mb=per_mb, obs_norm=obs_norm)
    doc = json.load(open(out))
    assert doc["status"] == "passed", doc
    for k in ("stock_vs_stock", "stock_vs_sim2gate"):
        assert doc[k]["differing"] == 0 and doc[k]["keys_match"] and not doc[k]["scalars_differing"]
    assert doc["stock_vs_sim2gate"]["tensors"] > 10 and doc["stock_vs_sim2gate"]["scalars_compared"] >= 3
    assert doc["monitor_records_written"] == 1 and doc["normalize_advantage_per_mini_batch"] == per_mb


def test_training_continues_exactly_as_stock(tmp_path):
    """The replays restore the state they started from: the run's final parameters equal a plain stock run's."""
    assert torch.equal(_train(tmp_path / "ui.json", iterations=3), _train(None, iterations=3))


def test_a_monitor_that_changes_training_fails(tmp_path, monkeypatch):
    class Meddling(TrainingMonitor):
        def __init__(self, runner, path, **k):
            super().__init__(runner, path, **k)
            orig = runner.alg.update

            def update(*a, **kw):
                with torch.no_grad():
                    next(runner.alg.actor.parameters()).add_(1e-6)    # a tiny side effect
                return orig(*a, **kw)
            runner.alg.update = update
            self._restore.append(lambda: setattr(runner.alg, "update", orig))
    monkeypatch.setattr("sim2gate.training.monitor.TrainingMonitor", Meddling)
    out = tmp_path / "ui.json"
    _train(out)
    doc = json.load(open(out))
    assert doc["status"] == "failed" and doc["stock_vs_sim2gate"]["differing"] > 0
    assert doc["stock_vs_stock"]["differing"] == 0


def test_a_nondeterministic_replay_is_inconclusive(tmp_path, monkeypatch):
    calls = {"n": 0}
    orig = ui.outcome

    def noisy(alg, loss):
        t, s = orig(alg, loss)
        calls["n"] += 1
        s["loss/noise"] = float(calls["n"])                       # every replay differs, like a nondeterministic GPU
        return t, s
    monkeypatch.setattr(ui, "outcome", noisy)
    out = tmp_path / "ui.json"
    _train(out)
    doc = json.load(open(out))
    assert doc["status"] == "inconclusive" and "already differs" in doc["note"]


def test_launcher_flag_requires_eta_zero(monkeypatch):
    from sim2gate.training import isaaclab_launch as il
    with pytest.raises(SystemExit):
        il.main(["--update-identity", "x.json", "--eta", "1", "--", "--task", "T"])
