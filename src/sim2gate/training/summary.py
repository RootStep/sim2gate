"""Summarize a TrainingMonitor JSONL file: Experiment T's training-signal summary."""
import json
import math


def summarize(path, last_fraction=0.2):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    if not recs:
        raise ValueError(f"{path}: no records")
    k = max(1, math.ceil(len(recs) * last_fraction))
    tail = recs[-k:]
    # statistics over the groups RSL-RL actually normalizes: the rollout, or each minibatch in per-minibatch mode
    groups = [g for r in tail for g in (r["signals"].get("groups") or
              [{"rho": r["signals"]["rho"], "regime": r["signals"]["regime"]}])]
    rhos = [g["rho"] for g in groups if g["rho"] is not None]
    regimes = [g["regime"] for g in groups]
    shares = {}
    for r in tail:
        for name, s in r["reward_term_abs_share"].items():
            if s is not None:
                shares.setdefault(name, []).append(s)
    return {
        "file": str(path), "iterations": len(recs), "window": k, "eta": recs[-1]["eta"],
        "normalization_group": recs[-1]["signals"].get("normalization_group", "rollout"), "groups": len(groups),
        "eta_is_training_eta": recs[-1]["eta_is_training_eta"],
        "mean_one_minus_rho": (sum(1 - x for x in rhos) / len(rhos)) if rhos else None,
        "reversal_share": regimes.count("sign_flip") / len(regimes) if regimes else None,
        "cancelled_share": regimes.count("affine_cancelled") / len(regimes) if regimes else None,
        "mean_zero_tail_fraction": sum(r["signals"]["zero_tail_fraction"] for r in tail) / len(tail),
        "reward_term_abs_share": {n: sum(v) / len(v) for n, v in sorted(shares.items())},
    }
