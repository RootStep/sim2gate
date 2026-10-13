"""Dependency-free check of the torch signals against reference outputs from ppo-shaping-diagnostics 0.3.2,
committed in tests/fixtures/signals_reference.json.gz (regenerate with tests/fixtures/make_signals_reference.py)."""
import gzip, json, os
import pytest
import torch

from sim2gate.training.signals import (gae, reward_tail, shaped_rewards, recover_raw_rewards, tail_predictor,
                                       correlation, rollout_signals, recovered_zero_mask)

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
    """recover_raw_rewards snaps only float32 rounding residue (8 ulp of the larger operand) to zero."""
    v = torch.full((2, 1), 5.0); to = torch.tensor([[True], [False]])
    raw = torch.tensor([[1e-3], [0.0]])
    stored = raw + 0.99 * v * to
    rec = recover_raw_rewards(stored, v, to, 0.99)
    assert abs(float(rec[0, 0]) - 1e-3) < 1e-5 and float(rec[1, 0]) == 0.0


@pytest.mark.parametrize("r", [1e-7, 1e-12, -3e-9])
def test_tiny_reward_at_timeout_with_zero_value_is_kept_exactly(r):
    """Third review: with V = 0 there is nothing to undo, so even a reward far below 1e-6 must survive unchanged."""
    v = torch.zeros(1, 1); to = torch.tensor([[True]])
    stored = torch.tensor([[r]], dtype=torch.float32)
    rec = recover_raw_rewards(stored, v, to, 0.99)
    assert float(rec[0, 0]) == float(stored[0, 0]) != 0.0


@pytest.mark.parametrize("value", [0.37, 5.0, 123.456, -42.0, 1e4, 1.0 / 3.0, 7.77777])
def test_zero_reward_at_timeout_is_classified_zero(value):
    """A zero raw reward next to any bonus magnitude counts as zero for the zero tail, whatever residue the float32
    round trip leaves in the recovered value."""
    v = torch.full((1, 1), value, dtype=torch.float32); to = torch.tensor([[True]])
    stored = torch.zeros(1, 1) + 0.99 * v * to          # what RSL-RL stores for a zero reward at a timeout
    rec = recover_raw_rewards(stored, v, to, 0.99)
    assert bool(recovered_zero_mask(stored, v, to, 0.99, rec)[0, 0])


def test_fourth_review_case_recovered_value_is_not_erased():
    """Fourth review: V = 1, gamma = 0.99, genuine reward 2e-7. float32 storage leaves ~1.79e-7 after undoing the
    bonus. The recovered value is kept as is (never snapped); only the zero classification, used for the zero
    tail and labeled as tolerance-based, calls it zero, because float32 cannot tell it from rounding there."""
    v = torch.ones(1, 1); to = torch.tensor([[True]])
    stored = torch.tensor([[2e-7]], dtype=torch.float32) + 0.99 * v
    rec = recover_raw_rewards(stored, v, to, 0.99)
    assert float(rec[0, 0]) != 0.0 and abs(float(rec[0, 0]) - 2e-7) < 1e-7
    assert bool(recovered_zero_mask(stored, v, to, 0.99, rec)[0, 0])        # indistinguishable: classified zero


@pytest.mark.parametrize("r", [2e-7, 1e-12, -3e-9])
def test_captured_raw_rewards_are_exact_next_to_a_bonus(r):
    """With captured raw rewards (what the monitor passes), nothing is recovered: a tiny reward at a timeout
    with V = 1 counts, and the zero tail is not inflated."""
    T_, N = 3, 1
    v = torch.ones(T_, N); lv = torch.ones(N)
    d = torch.tensor([[False], [False], [True]]); to = torch.tensor([[False], [False], [True]])
    raw = torch.tensor([[0.0], [0.0], [r]], dtype=torch.float32)
    stored = raw + 0.99 * v * to
    cap = rollout_signals(stored, v, lv, d, to, eta=1.0, gamma=0.99, lam=0.95, raw_rewards=raw)
    rec = rollout_signals(stored, v, lv, d, to, eta=1.0, gamma=0.99, lam=0.95)
    assert cap.raw_reward_source == "captured" and rec.raw_reward_source == "recovered"
    assert cap.zero_tail_fraction == 0.0                 # every step has the tiny reward ahead of it
    assert rec.zero_tail_fraction == 1.0                 # recovered: below float32 resolution next to the bonus
    assert cap.mean_raw_reward == pytest.approx(r / 3, rel=1e-6)


def test_captured_raw_rewards_shape_is_checked():
    v = torch.ones(2, 2)
    with pytest.raises(ValueError, match="raw_rewards shape"):
        rollout_signals(v, v, torch.ones(2), v.bool(), v.bool(), eta=1.0, gamma=0.99, lam=0.95,
                        raw_rewards=torch.ones(3, 2))


def test_reward_representable_next_to_bonus_survives():
    """1e-3 next to a bonus of ~5 is far above float32 resolution there (~5e-7), so it must survive."""
    v = torch.full((1, 1), 5.0); to = torch.tensor([[True]])
    stored = torch.tensor([[1e-3]]) + 0.99 * v
    assert abs(float(recover_raw_rewards(stored, v, to, 0.99)[0, 0]) - 1e-3) < 1e-6
