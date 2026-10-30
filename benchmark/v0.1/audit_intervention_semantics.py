"""Gate 3: CPU-only intervention-semantics audit; no simulator, training, or GPU.

Run: PYTHONPATH=src python benchmark/v0.1/audit_intervention_semantics.py

This audit checks configuration and source-level ramp mathematics. It cannot prove
that delays were sampled in a live actuator or that a policy learns a flaw.
"""
import json
import math
from pathlib import Path

from sim2gate.mvp import tasks as mvp
from sim2gate.v2 import tasks as v2
from sim2gate.v2.phase2_arms import candidate, DYNAMIC_RATIO


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def run():
    baseline = v2.P0_V5_STATIC_FRICTION
    modified = candidate("Phi-slip-mild", "mild")["static_friction"]
    base_width, changed_width = baseline[1] - baseline[0], modified[1] - modified[0]
    base_mid, changed_mid = sum(baseline) / 2, sum(modified) / 2
    width_only = (base_mid - changed_width / 2, base_mid + changed_width / 2)
    base_delay = mvp.P0_V3_DELAY_STEPS
    ramp_rounds = mvp.P0_V4_DELAY_RAMP_ROUNDS
    physics_per_round = mvp.PHYSICS_STEPS_PER_ROUND
    delay_candidate = candidate("Phi-delay", "nodelay")
    ramp_start, ramp_end = (n * physics_per_round for n in ramp_rounds)

    def max_delay_at(steps, lo, hi):
        fraction = min(1.0, max(0.0, (steps - ramp_start) / (ramp_end - ramp_start)))
        return lo + round((hi - lo) * fraction)

    sample_rounds = [0, ramp_rounds[0], (ramp_rounds[0] + ramp_rounds[1]) // 2,
                     ramp_rounds[1], ramp_rounds[1] + 1]
    schedule = [{"round": n, "physics_steps": n * physics_per_round,
                 "H0_max_delay": max_delay_at(n * physics_per_round, *base_delay),
                 "D2_max_delay": max_delay_at(n * physics_per_round, 0, delay_candidate["max_delay_steps"])}
                for n in sample_rounds]
    checks = {
        "D1_dynamic_ratio_matches_baseline": all(close(d, s * DYNAMIC_RATIO) for s, d in
            zip(v2.P0_V5_STATIC_FRICTION, v2.P0_V5_DYNAMIC_FRICTION)),
        "D1_reduced_width": changed_width < base_width,
        "D1_shifted_midpoint": not close(base_mid, changed_mid),
        "D1_candidate_static_range_matches_expected": tuple(modified) == (0.3, 0.6),
        "D2_baseline_has_positive_delay": base_delay[1] > 0,
        "D2_candidate_sets_zero_delay": delay_candidate["max_delay_steps"] == 0,
        "D2_ramp_starts_before_it_ends": ramp_start < ramp_end,
        "D2_baseline_delay_zero_before_ramp": schedule[0]["H0_max_delay"] == 0,
        "D2_baseline_delay_positive_after_ramp": schedule[-1]["H0_max_delay"] > 0,
        "D2_zero_at_all_sampled_points": all(row["D2_max_delay"] == 0 for row in schedule),
    }
    result = {
        "gate": "CPU Gate 3: intervention semantics (source/config only)",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "D1": {"baseline_static": baseline, "candidate_static": modified,
               "baseline_width": base_width, "candidate_width": changed_width,
               "baseline_midpoint": base_mid, "candidate_midpoint": changed_mid,
               "width_only_comparator_same_midpoint": width_only,
               "interpretation": "lower-and-narrower, not width-only"},
        "D2": {"baseline_delay_steps": base_delay, "candidate_max_delay_steps": 0,
               "ramp_rounds": ramp_rounds, "physics_steps_per_round": physics_per_round,
               "sample_schedule": schedule,
               "interpretation": "baseline max delay is zero early and positive only after ramp; live sampling not verified"},
        "limits": ["No Isaac Lab configuration instantiation in this gate (covered separately by Gate 2)",
                   "No live actuator reset/sampling check", "No learned policy manifestation or diagnosis measured"],
    }
    out = Path("benchmark/v0.1/GATE3_SEMANTICS_REPORT.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved {out}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run())
