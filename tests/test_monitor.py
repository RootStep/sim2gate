"""TrainingMonitor on real rsl-rl-lib training (CPU, toy Mountain Car). Skipped without rsl-rl-lib."""
import contextlib, io, json
import pytest
import torch

rsl_rl = pytest.importorskip("rsl_rl")
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from sim2gate.training import TrainingMonitor, BSRSPPO  # noqa: E402
from mountain_car_env import VecMountainCar  # noqa: E402

# rho per rollout recorded by the RSL-RL acceptance gate (addendum A captures), computed by the validated package
GATE_RHO = {
    ("sparse", 3, 2.0, False): [0.9257803311568907, 0.6540071970582763],
    ("dense", 1, 1.0, False): [0.9999985044329288, 0.9998423354738699],
}
GATE_MB_RHO = {("sparse", 1, 1.0, True): [0.9639531762679535, 0.949968548772697, 0.971341651762114, 0.9754879049696695,
                                          0.8692907966102926, 0.8654681483449465, 0.8605002496380207, 0.8895331440715003]}


def cfg(alg_class, eta, per_mb):
    alg = {"class_name": alg_class, "num_learning_epochs": 2, "num_mini_batches": 4, "clip_param": 0.2, "gamma": 0.99,
           "lam": 0.95, "value_loss_coef": 1.0, "entropy_coef": 0.0, "learning_rate": 1e-3, "max_grad_norm": 1.0,
           "use_clipped_value_loss": True, "schedule": "adaptive", "desired_kl": 0.01,
           "normalize_advantage_per_mini_batch": per_mb}
    if alg_class is BSRSPPO:
        alg["eta"] = eta
    mlp = {"class_name": "MLPModel", "hidden_dims": [64, 64], "activation": "elu", "obs_normalization": False}
    actor = dict(mlp, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
    return {"num_steps_per_env": 48, "save_interval": 10**9, "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "algorithm": alg, "actor": actor, "critic": dict(mlp)}


def train(reward, seed, eta, per_mb, alg_class, monitor_path=None, iterations=2):
    torch.manual_seed(seed)
    env = VecMountainCar(num_envs=16, reward=reward, max_episode_length=64, seed=seed, init_velocity=True)
    with contextlib.redirect_stdout(io.StringIO()):
        runner = OnPolicyRunner(env, cfg(alg_class, eta, per_mb), log_dir=None, device="cpu")
        mon = TrainingMonitor(runner, monitor_path) if monitor_path else None
        runner.learn(iterations, init_at_random_ep_len=True)
    if mon:
        mon.close()
    a = runner.alg
    params = torch.cat([p.detach().reshape(-1) for p in list(a.actor.parameters()) + list(a.critic.parameters())])
    recs = [json.loads(l) for l in open(monitor_path)] if monitor_path else None
    return params, recs


def alg_for(eta):
    return BSRSPPO if eta != 0.0 else "rsl_rl.algorithms:PPO"


@pytest.mark.parametrize("run", [("sparse", 1, 0.0, False), ("sparse", 3, 2.0, False), ("sparse", 1, 1.0, True),
                                 ("dense", 1, 1.0, False)])
def test_monitor_does_not_change_training(tmp_path, run):
    reward, seed, eta, per_mb = run
    off, _ = train(reward, seed, eta, per_mb, alg_for(eta))
    on, recs = train(reward, seed, eta, per_mb, alg_for(eta), tmp_path / "m.jsonl")
    assert torch.equal(off, on)
    assert len(recs) == 2


@pytest.mark.parametrize("run", list(GATE_RHO))
def test_rollout_rho_matches_gate(tmp_path, run):
    reward, seed, eta, per_mb = run
    _, recs = train(reward, seed, eta, per_mb, alg_for(eta), tmp_path / "m.jsonl")
    got = [r["signals"]["rho"] for r in recs]
    assert all(abs(g - e) < 1e-6 for g, e in zip(got, GATE_RHO[run]))
    assert all(r["eta_is_training_eta"] for r in recs)
    assert all(abs(r["signals"]["predicted_rho"] - r["signals"]["rho"]) < 1e-6 for r in recs)


@pytest.mark.parametrize("run", list(GATE_MB_RHO))
def test_minibatch_rho_matches_gate(tmp_path, run):
    reward, seed, eta, per_mb = run
    _, recs = train(reward, seed, eta, per_mb, alg_for(eta), tmp_path / "m.jsonl")
    got = [x for r in recs for x in r["signals"]["minibatch_rho"]]
    assert len(got) == 8
    assert all(abs(g - e) < 1e-6 for g, e in zip(got, GATE_MB_RHO[run]))


def test_reward_terms_logged(tmp_path):
    _, recs = train("dense", 1, 1.0, False, BSRSPPO, tmp_path / "m.jsonl")
    for r in recs:
        assert set(r["reward_terms"]) == {"goal", "effort"}
        assert abs(sum(r["reward_term_abs_share"].values()) - 1.0) < 1e-9


def test_stock_ppo_gets_what_if_eta(tmp_path):
    _, recs = train("sparse", 1, 0.0, False, "rsl_rl.algorithms:PPO", tmp_path / "m.jsonl")
    assert all(r["eta"] == 1.0 and not r["eta_is_training_eta"] for r in recs)


def test_bsrsppo_eta0_equals_ppo():
    a, _ = train("sparse", 2, 0.0, False, "rsl_rl.algorithms:PPO")
    b, _ = train("sparse", 2, 0.0, False, BSRSPPO)
    assert torch.equal(a, b)


def test_signals_summary_cli(tmp_path, capsys):
    from sim2gate.cli import main
    _, recs = train("sparse", 3, 2.0, False, BSRSPPO, tmp_path / "m.jsonl")
    assert main(["signals", str(tmp_path / "m.jsonl"), "--last", "0.5"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["window"] == 1 and abs(out["mean_one_minus_rho"] - (1 - GATE_RHO[("sparse", 3, 2.0, False)][1])) < 1e-6
