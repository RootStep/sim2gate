"""Per-rollout training signals for same-critic BSRS in PPO, vectorized in torch.

All tensors are [T, N] (time, environment), RSL-RL conventions:
  * dones[t] = the episode ended at step t (termination or time limit);
  * time_outs[t] = it ended by the time limit; the stored reward then already contains gamma * V(s_t),
    and the successor value used for bootstrapping and shaping is V(s_t);
  * last_values = critic value of the observation after the last step.

The math mirrors ppo-shaping-diagnostics 0.3.2, which was checked against rsl-rl-lib 5.5.1 down to the
actor gradient (RSL-RL acceptance gate, Oct 2026). This module is the fast path for logging during training;
tests/test_signals.py checks it against that package.
"""
from dataclasses import dataclass, asdict
import torch

F32_EPS = float(torch.finfo(torch.float32).eps)


def recover_raw_rewards(stored, values, time_outs, gamma):
    """Undo RSL-RL's timeout bonus: raw = stored - gamma * V(s_t) on time-limit steps.
    The float32 residue left where the raw reward was 0 is snapped to 0, so zero tails stay zero."""
    to = time_outs.bool()
    raw = stored - gamma * values * to
    snap = to & (raw.abs() <= 8 * F32_EPS * stored.abs().clamp_min(1.0))
    return torch.where(snap, torch.zeros_like(raw), raw)


def _successor_values(values, last_values, dones, time_outs):
    """Value of the successor state used for bootstrapping/shaping, and the no-bootstrap mask."""
    to = time_outs.bool()
    nxt = torch.cat([values[1:], last_values.reshape(1, -1)], dim=0)
    nxt = torch.where(to, values, nxt)
    terminated = dones.bool() & ~to
    return nxt, terminated


def gae(rewards, values, last_values, dones, time_outs, *, gamma, lam):
    """GAE on RAW rewards with RSL-RL's timeout convention (bootstrap gamma * V(s_t) at time limits)."""
    nxt, terminated = _successor_values(values, last_values, dones, time_outs)
    boot = torch.where(terminated, torch.zeros_like(nxt), gamma * nxt)
    cont = (~dones.bool()).to(rewards.dtype)
    adv = torch.zeros_like(rewards)
    acc = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[t] + boot[t] - values[t]
        acc = delta + gamma * lam * cont[t] * acc
        adv[t] = acc
    return adv


def reward_tail(rewards, dones, *, gamma, lam):
    """T_t = sum_k (gamma*lam)^k r_{t+k} within the episode segment (GAE with V = 0)."""
    cont = (~dones.bool()).to(rewards.dtype)
    tail = torch.zeros_like(rewards)
    acc = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        acc = rewards[t] + gamma * lam * cont[t] * acc
        tail[t] = acc
    return tail


def shaped_rewards(rewards, values, last_values, dones, time_outs, *, eta, gamma):
    """Same-critic BSRS: r' = r + eta * (gamma * b * V_next - V), with V_next = V(s_t) on time limits, 0 on terminations."""
    nxt, terminated = _successor_values(values, last_values, dones, time_outs)
    phi_next = torch.where(terminated, torch.zeros_like(nxt), nxt)
    return rewards + eta * (gamma * phi_next - values)


def correlation(a, b):
    x = a.double().reshape(-1); y = b.double().reshape(-1)
    xc = x - x.mean(); yc = y - y.mean()
    den = xc.norm() * yc.norm()
    return None if float(den) == 0.0 else float(torch.clamp(xc @ yc / den, -1, 1))


def tail_predictor(a, t, eta, affine_tolerance=1e-6):
    """Regime and rho of same-critic shaping from (A, T) alone, for one normalization group.
    Same algebra and regime names as ppo_shaping_diagnostic.tail_predictor."""
    x = a.double().reshape(-1); y = t.double().reshape(-1)
    ac = x - x.mean(); tc = y - y.mean()
    na2 = float(ac @ ac)
    if na2 == 0.0:
        return {"beta": None, "scale": None, "residual_ratio": None, "rho": None, "regime": "degenerate"}
    na = na2 ** 0.5
    beta = float(tc @ ac) / na2
    nr = float((tc - beta * ac).norm())
    c = 1.0 + eta - eta * beta
    num = c * na; den = (num * num + (eta * nr) ** 2) ** 0.5
    if den == 0.0:
        return {"beta": beta, "scale": c, "residual_ratio": None, "rho": None, "regime": "degenerate"}
    rho = max(-1.0, min(1.0, num / den))
    q = float("inf") if c == 0 else abs(eta) * nr / (abs(c) * na)
    if c < 0: regime = "sign_flip"
    elif c == 0: regime = "orthogonal"
    elif q <= affine_tolerance: regime = "affine_cancelled"
    else: regime = "partial"
    return {"beta": beta, "scale": c, "residual_ratio": q, "rho": rho, "regime": regime}


def _rel_max_abs(x, ref):
    x = x.double().reshape(-1); ref = ref.double().reshape(-1)
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1.0))


def _normalize(x):
    """RSL-RL's advantage normalization: (x - mean) / (std + 1e-8), torch std with ddof 1."""
    return (x - x.mean()) / (x.std() + 1e-8)


@dataclass
class RolloutSignals:
    eta: float
    normalization_group: str          # "rollout" or "minibatch": the groups RSL-RL normalizes advantages over
    num_samples: int
    rho: float | None                 # rollout-level corr(A, A') (the statistic itself only in rollout mode)
    predicted_rho: float | None       # rollout-level rho predicted from (A, T) alone, before A' exists
    regime: str                       # rollout-level regime
    beta: float | None
    residual_ratio: float | None
    zero_tail_fraction: float
    identity_residual: float          # max |A' - ((1+eta) A - eta T)|, a coupling sanity check
    mean_raw_reward: float
    timeouts: int
    terminations: int
    minibatch_rho: list | None = None # per minibatch, when advantages are normalized per minibatch
    groups: list | None = None        # one {rho, predicted_rho, regime} per actual normalization group
    trainer_signal_residual: float | None = None         # reconstruction vs the trainer's own returns - values
    trainer_normalization_residual: float | None = None  # normalized reconstruction vs advantages trained on
    minibatch_capture: str | None = None                 # "ok", or why minibatch groups are unavailable

    def as_dict(self):
        return asdict(self)


def rollout_signals(stored_rewards, values, last_values, dones, time_outs, *, eta, gamma, lam,
                    minibatch_indices=None, trainer_advantages=None, trainer_normalized=None,
                    training_eta=False, minibatch_capture=None):
    """All per-rollout signals from RSL-RL storage tensors ([T, N] or [T, N, 1]).

    minibatch_indices: list of 1-D index tensors into the time-major flattened rollout (t * N + e), when RSL-RL
    normalizes advantages per minibatch; the per-group statistics are then computed for each minibatch.
    trainer_advantages: the trainer's own unnormalized advantages (storage returns - values) after
    compute_returns; compared with the reconstructed A' if training_eta else A, so a trainer whose shaping differs
    from the claimed eta is caught. trainer_normalized: the advantages the trainer will train on (rollout mode)."""
    sq = lambda x: x[..., 0] if x.dim() == 3 else x
    r_st, v, d, to = sq(stored_rewards), sq(values), sq(dones).bool(), sq(time_outs).bool()
    lv = last_values.reshape(-1).to(v.dtype)
    raw = recover_raw_rewards(r_st, v, to, gamma)
    A = gae(raw, v, lv, d, to, gamma=gamma, lam=lam)
    T = reward_tail(raw, d, gamma=gamma, lam=lam)
    Ap = gae(shaped_rewards(raw, v, lv, d, to, eta=eta, gamma=gamma), v, lv, d, to, gamma=gamma, lam=lam)
    ident = float((Ap.double() - ((1 + eta) * A.double() - eta * T.double())).abs().max())
    tp = tail_predictor(A, T, eta)
    trained = Ap if training_eta else A
    sig_res = None if trainer_advantages is None else _rel_max_abs(sq(trainer_advantages), trained)
    norm_res = None
    if trainer_normalized is not None and minibatch_indices is None:
        norm_res = _rel_max_abs(sq(trainer_normalized), _normalize(trained))
    groups, mb = None, None
    if minibatch_indices is not None:
        Af, Apf, Tf = A.reshape(-1), Ap.reshape(-1), T.reshape(-1)
        groups = []
        for ix in minibatch_indices:
            g = tail_predictor(Af[ix], Tf[ix], eta)
            groups.append({"rho": correlation(Af[ix], Apf[ix]), "predicted_rho": g["rho"], "regime": g["regime"]})
        mb = [g["rho"] for g in groups]
    else:
        groups = [{"rho": correlation(A, Ap), "predicted_rho": tp["rho"], "regime": tp["regime"]}]
    return RolloutSignals(
        eta=float(eta), normalization_group="minibatch" if minibatch_indices is not None else "rollout",
        num_samples=int(A.numel()), rho=correlation(A, Ap), predicted_rho=tp["rho"], regime=tp["regime"],
        beta=tp["beta"], residual_ratio=tp["residual_ratio"], zero_tail_fraction=float((T == 0).double().mean()),
        identity_residual=ident, mean_raw_reward=float(raw.double().mean()),
        timeouts=int(to.sum()), terminations=int((d & ~to).sum()), minibatch_rho=mb, groups=groups,
        trainer_signal_residual=sig_res, trainer_normalization_residual=norm_res,
        minibatch_capture=minibatch_capture)
