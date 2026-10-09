"""Summarize a TrainingMonitor JSONL file: Experiment T's training-signal summary."""
import json
import math


def summarize(path, last_fraction=0.2):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    if not recs:
        raise ValueError(f"{path}: no records")
    k = max(1, math.ceil(len(recs) * last_fraction))
    tail = recs[-k:]
    rhos = [r["signals"]["rho"] for r in tail if r["signals"]["rho"] is not None]
    regimes = [r["signals"]["regime"] for r in tail]
    shares = {}
    for r in tail:
        for name, s in r["reward_term_abs_share"].items():
            if s is not None:
                shares.setdefault(name, []).append(s)
    return {
        "file": str(path), "iterations": len(recs), "window": k, "eta": recs[-1]["eta"],
        "eta_is_training_eta": recs[-1]["eta_is_training_eta"],
        "mean_one_minus_rho": (sum(1 - x for x in rhos) / len(rhos)) if rhos else None,
        "reversal_share": regimes.count("sign_flip") / len(regimes),
        "cancelled_share": regimes.count("affine_cancelled") / len(regimes),
        "mean_zero_tail_fraction": sum(r["signals"]["zero_tail_fraction"] for r in tail) / len(tail),
        "reward_term_abs_share": {n: sum(v) / len(v) for n, v in sorted(shares.items())},
    }
