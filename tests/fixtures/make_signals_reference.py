"""Regenerate signals_reference.json.gz from ppo-shaping-diagnostics 0.3.2 (the package validated against CleanRL,
SB3 and rsl-rl-lib). Needs that package; the committed fixture lets CI check against it without the package.

    python tests/fixtures/make_signals_reference.py
"""
import gzip, json, os, sys
import numpy as np
import ppo_shaping_diagnostic as psd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_signals import make_rollout, pack_rslrl, GAMMA, LAM  # noqa: E402

CONV = psd.PPOConventions(dtype="float32", ddof=1, eps=1e-8, clip_coef=0.2, bootstrap_truncation=True,
                          objective="loss", adam_eps=1e-8, clip_norm_eps=1e-6)
cases = []
for seed in (0, 1):
    for sparse in (True, False):
        stored, raw, v, lv, d, to = make_rollout(seed, sparse)
        pk = pack_rslrl(raw.numpy(), v.numpy(), d.numpy(), to.numpy(), lv.numpy())
        links = [pk[k] for k in ("values", "next_values", "terminated", "truncated", "segment_end")]
        kw = dict(gamma=GAMMA, lam=LAM, conventions=CONV)
        A = psd.gae(pk["rewards"], *links, **kw); T = psd.reward_tail(pk["rewards"], *links[2:], **kw)
        case = {"seed": seed, "sparse": sparse, "shape": list(raw.shape),
                "inputs": {k: x.reshape(-1).tolist() for k, x in
                           dict(stored=stored, values=v, last_values=lv, dones=d.float(), time_outs=to.float()).items()},
                "A": A.tolist(), "T": T.tolist(), "by_eta": {}}
        for eta in (0.5, 1.0, 2.0):
            Ap = psd.gae(psd.bsrs_shaped_rewards(pk["rewards"], *links, eta=eta, gamma=GAMMA, conventions=CONV),
                         *links, **kw)
            tp = psd.tail_predictor(A, T, eta)
            case["by_eta"][str(eta)] = {"Ap": Ap.tolist(), "rho": psd.correlation_predictor(A, Ap),
                                        "predicted_rho": tp["rho"], "regime": tp["regime"]}
        cases.append(case)
out = os.path.join(os.path.dirname(__file__), "signals_reference.json.gz")
with gzip.open(out, "wt") as f:
    json.dump({"generator": f"ppo-shaping-diagnostics {psd.__version__}", "gamma": GAMMA, "lam": LAM,
               "packing": "env-major", "cases": cases}, f)
print("wrote", out)
