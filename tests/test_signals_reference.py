"""Dependency-free check of the torch signals against reference outputs from ppo-shaping-diagnostics 0.3.2,
committed in tests/fixtures/signals_reference.json.gz (regenerate with tests/fixtures/make_signals_reference.py)."""
import gzip, json, os
import pytest
import torch

from sim2gate.training.signals import (gae, reward_tail, shaped_rewards, recover_raw_rewards, tail_predictor,
                                       correlation, rollout_signals)

REF = json.load(gzip.open(os.path.join(os.path.dirname(__file__), "fixtures", "signals_reference.json.gz"), "rt"))
G, L = REF["gamma"], REF["lam"]


def env_major(x):
    return x.t().reshape(-1)          # [T, N] -> packed env-major, the package's layout


def close(x, ref, tol=1e-5):
    ref = torch.tensor(ref, dtype=torch.float64)
    return float((x.double() - ref).abs().max()) <= tol * max(1.0, float(ref.abs().max()))


@pytest.mark.parametrize("case", REF["cases"], ids=lambda c: f"seed{c['seed']}-{'sparse' if c['sparse'] else 'dense'}")
def test_signals_match_reference(case):
    T_, N = case["shape"]
    inp = {k: torch.tensor(v, dtype=torch.float32) for k, v in case["inputs"].items()}
    stored, v = inp["stored"].reshape(T_, N), inp["values"].reshape(T_, N)
    lv, d, to = inp["last_values"], inp["dones"].reshape(T_, N).bool(), inp["time_outs"].reshape(T_, N).bool()
    raw = recover_raw_rewards(stored, v, to, G)
    A = gae(raw, v, lv, d, to, gamma=G, lam=L)
    T = reward_tail(raw, d, gamma=G, lam=L)
    assert close(env_major(A), case["A"]) and close(env_major(T), case["T"])
    for eta_s, ref in case["by_eta"].items():
        eta = float(eta_s)
        Ap = gae(shaped_rewards(raw, v, lv, d, to, eta=eta, gamma=G), v, lv, d, to, gamma=G, lam=L)
        assert close(env_major(Ap), ref["Ap"])
        tp = tail_predictor(A, T, eta)
        assert tp["regime"] == ref["regime"]
        if ref["rho"] is not None:
            assert abs(correlation(A, Ap) - ref["rho"]) < 1e-6
            assert abs(tp["rho"] - ref["predicted_rho"]) < 1e-6


def test_zero_tail_cancels():
    case = REF["cases"][0]
    T_, N = case["shape"]
    inp = {k: torch.tensor(v, dtype=torch.float32) for k, v in case["inputs"].items()}
    v, lv = inp["values"].reshape(T_, N), inp["last_values"]
    d, to = inp["dones"].reshape(T_, N).bool(), inp["time_outs"].reshape(T_, N).bool()
    stored = 0.99 * v * to                         # no reward anywhere: only RSL-RL's timeout bonus, so T = 0
    sig = rollout_signals(stored, v, lv, d, to, eta=1.0, gamma=G, lam=L)
    assert sig.zero_tail_fraction == 1.0
    assert sig.regime == "affine_cancelled" and abs(sig.rho - 1.0) < 1e-9


def test_tiny_real_reward_on_timeout_step_survives():
    """recover_raw_rewards snaps only float32 rounding residue (8 ulp of the stored value) to zero."""
    v = torch.full((2, 1), 5.0); to = torch.tensor([[True], [False]])
    raw = torch.tensor([[1e-3], [0.0]])
    stored = raw + 0.99 * v * to
    rec = recover_raw_rewards(stored, v, to, 0.99)
    assert abs(float(rec[0, 0]) - 1e-3) < 1e-5 and float(rec[1, 0]) == 0.0
