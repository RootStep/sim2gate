"""The fast torch signals agree with ppo-shaping-diagnostics 0.3.2, the package validated against rsl-rl-lib.
Skipped when that package is not installed (it is not on PyPI yet)."""
import numpy as np
import pytest
import torch

from sim2gate.training.signals import (gae, reward_tail, shaped_rewards, recover_raw_rewards, tail_predictor,
                                       rollout_signals)

psd = pytest.importorskip("ppo_shaping_diagnostic")
from ppo_shaping_diagnostic.adapters import pack_time_env  # noqa: E402

GAMMA, LAM = 0.99, 0.95


def pack_rslrl(raw, values, dones, time_outs, last_values):
    """Package-side packing with RSL-RL conventions (same as the gate's rslrl_adapter.pack_rslrl)."""
    T, N = raw.shape
    out = {k: [] for k in ("rewards", "values", "next_values", "terminated", "truncated", "segment_end")}
    for e in range(N):
        nv = np.r_[values[1:, e], last_values[e]]
        nv = np.where(time_outs[:, e], values[:, e], nv)
        seg = np.zeros(T, bool); seg[-1] = not dones[-1, e]
        out["rewards"].append(raw[:, e]); out["values"].append(values[:, e]); out["next_values"].append(nv)
        out["terminated"].append(dones[:, e] & ~time_outs[:, e]); out["truncated"].append(time_outs[:, e])
        out["segment_end"].append(seg)
    return {k: np.concatenate(x) for k, x in out.items()}


def make_rollout(seed, sparse):
    g = torch.Generator().manual_seed(seed)
    T, N = 24, 32
    dones = torch.rand(T, N, generator=g) < 0.08
    time_outs = dones & (torch.rand(T, N, generator=g) < 0.5)
    if sparse:
        raw = (torch.rand(T, N, generator=g) < 0.05).float() * (dones & ~time_outs).float()
    else:
        raw = torch.randn(T, N, generator=g)
    values = torch.randn(T, N, generator=g) * 2
    last_values = torch.randn(N, generator=g) * 2
    stored = raw + GAMMA * values * time_outs        # what RSL-RL stores
    return stored, raw, values, last_values, dones, time_outs


def agree(x, ref, tol=1e-5):
    x = np.asarray(x, np.float64); ref = np.asarray(ref, np.float64)
    return float(np.max(np.abs(x - ref))) <= tol * max(1.0, float(np.max(np.abs(ref))))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("eta", [0.5, 1.0, 2.0])
def test_against_package(seed, sparse, eta):
    stored, raw, v, lv, d, to = make_rollout(seed, sparse)
    rec = recover_raw_rewards(stored, v, to, GAMMA)
    assert agree(rec, raw)
    conv = psd.PPOConventions(dtype="float32", ddof=1, eps=1e-8, clip_coef=0.2, bootstrap_truncation=True,
                              objective="loss", adam_eps=1e-8, clip_norm_eps=1e-6)
    pk = pack_rslrl(raw.numpy(), v.numpy(), d.numpy(), to.numpy(), lv.numpy())
    links = [pk[k] for k in ("values", "next_values", "terminated", "truncated", "segment_end")]
    kw = dict(gamma=GAMMA, lam=LAM, conventions=conv)
    A_ref = psd.gae(pk["rewards"], *links, **kw)
    T_ref = psd.reward_tail(pk["rewards"], *links[2:], **kw)
    Ap_ref = psd.gae(psd.bsrs_shaped_rewards(pk["rewards"], *links, eta=eta, gamma=GAMMA, conventions=conv), *links, **kw)
    A = gae(rec, v, lv, d, to, gamma=GAMMA, lam=LAM)
    T = reward_tail(rec, d, gamma=GAMMA, lam=LAM)
    Ap = gae(shaped_rewards(rec, v, lv, d, to, eta=eta, gamma=GAMMA), v, lv, d, to, gamma=GAMMA, lam=LAM)
    assert agree(pack_time_env(A.numpy()), A_ref)
    assert agree(pack_time_env(T.numpy()), T_ref)
    assert agree(pack_time_env(Ap.numpy()), Ap_ref)
    mine, ref = tail_predictor(A, T, eta), psd.tail_predictor(A_ref, T_ref, eta)
    assert mine["regime"] == ref["regime"]
    if ref["rho"] is not None:
        assert abs(mine["rho"] - ref["rho"]) < 1e-6
    sig = rollout_signals(stored, v, lv, d, to, eta=eta, gamma=GAMMA, lam=LAM)
    assert abs(sig.rho - psd.correlation_predictor(A_ref, Ap_ref)) < 1e-6
    assert sig.identity_residual <= 1e-4 * (1 + eta) * max(1.0, float(np.abs(A_ref).max()), float(np.abs(T_ref).max()))
