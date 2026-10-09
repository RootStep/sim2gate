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


@dataclass
class RolloutSignals:
    eta: float
    normalization_group: str          # "rollout" or "minibatch"
    num_samples: int
    rho: float | None                 # corr(A, A') over the normalization group (rollout mode)
    predicted_rho: float | None       # from (A, T) alone, before A' exists
    regime: str
    beta: float | None
    residual_ratio: float | None
    zero_tail_fraction: float
    identity_residual: float          # max |A' - ((1+eta) A - eta T)|, a coupling sanity check
    mean_raw_reward: float
    timeouts: int
    terminations: int
    minibatch_rho: list | None = None # per minibatch, when advantages are normalized per minibatch

    def as_dict(self):
        return asdict(self)


def rollout_signals(stored_rewards, values, last_values, dones, time_outs, *, eta, gamma, lam,
                    minibatch_indices=None):
    """All per-rollout signals from RSL-RL storage tensors ([T, N] or [T, N, 1]).

    minibatch_indices: optional list of 1-D index tensors into the time-major flattened rollout
    (t * N + e), used when RSL-RL normalizes advantages per minibatch."""
    sq = lambda x: x[..., 0] if x.dim() == 3 else x
    r_st, v, d, to = sq(stored_rewards), sq(values), sq(dones).bool(), sq(time_outs).bool()
    lv = last_values.reshape(-1).to(v.dtype)
    raw = recover_raw_rewards(r_st, v, to, gamma)
    A = gae(raw, v, lv, d, to, gamma=gamma, lam=lam)
    T = reward_tail(raw, d, gamma=gamma, lam=lam)
    Ap = gae(shaped_rewards(raw, v, lv, d, to, eta=eta, gamma=gamma), v, lv, d, to, gamma=gamma, lam=lam)
    ident = float((Ap.double() - ((1 + eta) * A.double() - eta * T.double())).abs().max())
    tp = tail_predictor(A, T, eta)
    mb = None
    if minibatch_indices is not None:
        Af, Apf = A.reshape(-1), Ap.reshape(-1)
        mb = [correlation(Af[ix], Apf[ix]) for ix in minibatch_indices]
    return RolloutSignals(
        eta=float(eta), normalization_group="minibatch" if minibatch_indices is not None else "rollout",
        num_samples=int(A.numel()), rho=correlation(A, Ap), predicted_rho=tp["rho"], regime=tp["regime"],
        beta=tp["beta"], residual_ratio=tp["residual_ratio"], zero_tail_fraction=float((T == 0).double().mean()),
        identity_residual=ident, mean_raw_reward=float(raw.double().mean()),
        timeouts=int(to.sum()), terminations=int((d & ~to).sum()), minibatch_rho=mb)
