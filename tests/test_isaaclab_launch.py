"""The launcher's patch path, exercised with a stand-in for Isaac Lab's train.py (no Isaac Lab needed)."""
import json
import textwrap
import pytest

pytest.importorskip("rsl_rl")
from sim2gate.training import isaaclab_launch  # noqa: E402

FAKE_TRAIN = textwrap.dedent('''
    import sys, contextlib, io, torch
    from rsl_rl.runners import OnPolicyRunner          # imported the way Isaac Lab does, at module top
    import cli_args                                    # sibling import, like Isaac Lab's train.py
    from mountain_car_env import VecMountainCar
    assert sys.argv[1:] == ["--task", "Fake-v0"], sys.argv
    torch.manual_seed(1)
    env = VecMountainCar(num_envs=16, reward="sparse", max_episode_length=64, seed=1, init_velocity=True)
    mlp = {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "elu", "obs_normalization": False}
    cfg = {"num_steps_per_env": 48, "save_interval": 10**9, "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
           "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 4, "entropy_coef": 0.0},
           "actor": dict(mlp, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0}),
           "critic": dict(mlp)}
    with contextlib.redirect_stdout(io.StringIO()):
        runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
        runner.learn(3, init_at_random_ep_len=True)
    open(sys.argv[0] + ".alg", "w").write(type(runner.alg).__name__ + " " + str(getattr(runner.alg, "eta", None)))
''')


@pytest.mark.parametrize("eta", [0.0, 2.0])
def test_launcher_with_script_entry_point(tmp_path, eta):
    (tmp_path / "cli_args.py").write_text("")
    script = tmp_path / "train.py"; script.write_text(FAKE_TRAIN)
    out = tmp_path / "signals.jsonl"
    from rsl_rl.runners import OnPolicyRunner
    orig = OnPolicyRunner.__init__
    try:
        isaaclab_launch.main(["--eta", str(eta), "--signals", str(out), "--isaaclab-script", str(script),
                              "--", "--task", "Fake-v0"])
    finally:
        OnPolicyRunner.__init__ = orig
    alg = (tmp_path / "train.py.alg").read_text().split()
    assert alg[0] == ("BSRSPPO" if eta else "PPO")
    recs = [json.loads(l) for l in open(out)]
    assert len(recs) == 3
    assert all(r["eta"] == (eta if eta else 1.0) and r["eta_is_training_eta"] == bool(eta) for r in recs)
