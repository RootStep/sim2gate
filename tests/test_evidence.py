"""EvidenceRecorder (v2 phase 0A) on real rsl-rl-lib training (CPU, toy Mountain Car with two reward terms)."""
import contextlib, copy, io, json
import numpy as np
import pytest
import torch

rsl_rl = pytest.importorskip("rsl_rl")
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from sim2gate.training import TrainingMonitor, BSRSPPO  # noqa: E402
from sim2gate.training.evidence import EvidenceRecorder  # noqa: E402
from sim2gate.training.signals import gae, reward_tail  # noqa: E402
from mountain_car_env import VecMountainCar  # noqa: E402
from test_monitor import cfg, alg_for  # noqa: E402

ITERATIONS = 3


class ToyTerms:
    def __init__(self, env):
        self.env, self.names, self.dt = env, list(env.reward_terms), 1.0

    def step_terms(self):
        return self.env.last_terms


def train(seed, eta, out=None, refit=False, monitor=True, sample_envs=16, obs_norm=True):
    torch.manual_seed(seed)
    env = VecMountainCar(num_envs=16, reward="dense", max_episode_length=64, seed=seed, init_velocity=True)
    with contextlib.redirect_stdout(io.StringIO()):
        runner = OnPolicyRunner(env, cfg(alg_for(eta), eta, False, obs_norm=obs_norm), log_dir=None, device="cpu")
        mon = TrainingMonitor(runner, out / "monitor.jsonl") if (out and monitor) else None
        rec = None
        if out:
            rec = EvidenceRecorder(runner, str(out), terms=ToyTerms(env), deep=range(ITERATIONS), grad_save=(0, 2),
                                   sample_envs=sample_envs, refit_terms=("effort",) if refit else (),
                                   refit_iterations=(1,) if refit else ())
        runner.learn(ITERATIONS, init_at_random_ep_len=True)
    for m in (rec, mon):
        if m:
            m.close()
    a = runner.alg
    params = torch.cat([p.detach().reshape(-1) for p in list(a.actor.parameters()) + list(a.critic.parameters())])
    recs = [json.loads(l) for l in open(out / "updates.jsonl")] if out else None
    return params, recs, runner


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_recorder_does_not_change_training(tmp_path, eta):
    off, _, _ = train(1, eta)
    on, recs, _ = train(1, eta, tmp_path, refit=True)
    assert torch.equal(off, on)
    assert [r["iteration"] for r in recs] == list(range(ITERATIONS))


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_identities_hold(tmp_path, eta):
    _, recs, _ = train(2, eta, tmp_path)
    for r in recs:
        assert r["term_sum_residual"] < 1e-5            # the terms add up to the reward the trainer got
        assert r["identity_residual"] < 1e-3            # A = sum_k T_k + V (float32 GAE, rewards up to 100)
        assert r["trainer_residual"] < 1e-5             # our A (or BSRS A') is what the trainer computed
        assert abs(r["direction_cos_trainer"]["trainer"] - 1.0) < 1e-9
        if eta == 0.0:                                  # unshaped training: trainer direction = normalized A
            assert r["direction_cos_trainer"]["A"] > 1 - 1e-5
        m = np.array(r["moments"]["cov"])
        assert m.shape == (4, 4) and np.allclose(m, m.T)
        assert -1 <= r["bsrs_what_if_rho"]["1"] <= 1
        assert set(r["update_cos"]) == set(r["direction_cos_trainer"])
        assert r["kl_subset"] >= 0 and 0 <= r["clip_fraction_subset"] <= 1
    if eta == 1.0:   # the what-if rho at the training eta is the monitor's (gated) rho
        mon = [json.loads(l) for l in open(tmp_path / "monitor.jsonl")]
        for r, m in zip(recs, mon):
            assert abs(r["bsrs_what_if_rho"]["1"] - m["signals"]["rho"]) < 1e-6


def test_files_and_offline_recomputation(tmp_path):
    """Everything in the sample dump reproduces the recorded quantities offline, from the saved models alone."""
    # observation normalization off, as in the Go2 task: with it on, rsl-rl < 5.5 updates the normalizer during the
    # rollout, so log-probs at collection differ slightly from the pre-update actor's (a trainer fact, not ours)
    _, recs, runner = train(3, 0.0, tmp_path, refit=True, obs_norm=False)
    meta = json.load(open(tmp_path / "meta.json"))
    assert meta["reward_terms"] == ["goal", "effort"] and meta["moments_columns"] == ["A", "T:goal", "T:effort", "V"]
    assert "refit" in recs[1] and recs[1]["refit"]["effort"]["value_loss_by_round"]
    assert "refit_minus:effort" in recs[1]["direction_cos_trainer"]
    for it in range(ITERATIONS):
        assert (tmp_path / "deep" / f"it{it:05d}_samples.npz").exists()
    assert sorted(p.name for p in (tmp_path / "deep").glob("*_directions.npz")) == \
        ["it00000_directions.npz", "it00002_directions.npz"]

    it = 2
    s = np.load(tmp_path / "deep" / f"it{it:05d}_samples.npz")
    t = lambda k: torch.from_numpy(s[k])  # noqa: E731
    gamma, lam = meta["gamma"], meta["lam"]
    d, to, v, lv = t("dones").bool(), t("time_outs").bool(), t("values"), t("last_values")
    A = gae(t("raw_rewards"), v, lv, d, to, gamma=gamma, lam=lam)
    assert torch.allclose(A, t("trainer_advantages_unnormalized"), atol=1e-4)
    Tk = torch.stack([reward_tail(t("terms")[..., k], d, gamma=gamma, lam=lam) for k in range(2)], -1)
    V0 = gae(torch.zeros_like(A), v, lv, d, to, gamma=gamma, lam=lam)
    assert torch.allclose(A, Tk.sum(-1) + V0, atol=1e-3)
    assert torch.allclose(t("terms").sum(-1), t("raw_rewards"), atol=1e-5)
    # the stored (bonus-included) rewards are raw + gamma V on time-outs, as RSL-RL stores them
    assert torch.allclose(t("stored_rewards"), t("raw_rewards") + gamma * v * to, atol=1e-4)

    # the saved trainer direction = -(gradient of PPO's surrogate at ratio 1), recomputed from saved models
    actor = copy.deepcopy(runner.alg.actor)
    actor.load_state_dict(torch.load(tmp_path / "deep" / f"it{it:05d}_models.pt")["actor"])
    from tensordict import TensorDict
    obs = TensorDict({"policy": t("obs_policy")}, batch_size=t("obs_policy").shape[:2]).flatten(0, 1)
    actions = t("actions").flatten(0, 1)
    old_logp = t("old_log_prob").reshape(-1)
    adv = t("trainer_advantages").reshape(-1)
    actor.distribution.update(actor.mlp(actor.get_latent(obs)))
    logp = actor.get_output_log_prob(actions)
    assert torch.allclose(logp.detach(), old_logp, atol=1e-5)   # pre-update actor reproduces the rollout
    ratio = torch.exp(logp - old_logp)
    surrogate = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 0.8, 1.2)).mean()
    params = [q for q in actor.parameters() if q.requires_grad]
    g = -torch.cat([x.reshape(-1) for x in torch.autograd.grad(surrogate, params)])
    dz = np.load(tmp_path / "deep" / f"it{it:05d}_directions.npz")
    saved = torch.from_numpy(dz["directions"][list(dz["names"]).index("trainer")]).double()
    assert torch.allclose(g.double(), saved, rtol=1e-4, atol=1e-7)
    assert dz["actual_update"].shape == saved.shape
    # cfg(): 2 epochs x 4 minibatches; the first minibatch step starts at the pre-update actor (ratio 1), so its
    # actor gradient is the surrogate's on that minibatch (plus the entropy term, coefficient 0 here)
    mg = dz["minibatch_loss_grads"]
    assert mg.shape == (8, saved.numel())
    assert len(recs[2]["minibatch_grad_norm"]) == 8


def test_rejects_unknown_refit_term(tmp_path):
    torch.manual_seed(0)
    env = VecMountainCar(num_envs=4, reward="dense", seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        runner = OnPolicyRunner(env, cfg(alg_for(0.0), 0.0, False), log_dir=None, device="cpu")
    with pytest.raises(ValueError):
        EvidenceRecorder(runner, str(tmp_path), terms=ToyTerms(env), refit_terms=("nope",), refit_iterations=(0,))
