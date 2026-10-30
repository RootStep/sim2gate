"""Strict, label-free adapter for existing Sim2Gate Check 2 results.

Runs the actual production score() first; exports only an allowlisted verdict.
This is a development contract, not a frozen v0.1 scientific method.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sim2gate.checks.exploits_score import score as production_check2_score

METRICS = frozenset((
    "foot_slip_p95_mps", "ground_penetration_m", "contact_force_p99_bw",
    "action_jitter_high_freq_power_share", "actuator_limit_step_share",
))
ALLOWED = frozenset(("opaque_id", "status", "flagged", "metrics"))
SCHEMA = "v0.1-check2-sanitized-draft"


def export_check2(opaque_id, raw_report, calibration):
    if not isinstance(opaque_id, str) or not opaque_id or len(opaque_id) > 80:
        raise ValueError("invalid opaque ID")
    # Deliberately do not serialize the production result: it includes checkpoint
    # SHA and free-text reasons / advice that may contain sensitive metadata.
    original = production_check2_score(raw_report, calibration)
    status = original["status"]
    if status not in ("scored", "invalid"):
        raise ValueError("unknown check status")
    if set(original["metrics"]) != METRICS:
        raise ValueError("unexpected production metrics")
    metrics = {}
    for name in sorted(METRICS):
        item = original["metrics"][name]
        # Expose only boolean flags when applicability checks passed.
        metrics[name] = item["flagged"] if status == "scored" else None
    return {"opaque_id": opaque_id, "status": status,
            "flagged": original["flagged"] if status == "scored" else None,
            "metrics": metrics}


def validate_sanitized(records):
    if not isinstance(records, list) or not records:
        raise ValueError("nonempty list required")
    ids = set()
    for rec in records:
        if not isinstance(rec, dict) or set(rec) != ALLOWED:
            raise ValueError("unexpected sanitized fields")
        rid = rec["opaque_id"]
        if not isinstance(rid, str) or not rid or len(rid) > 80 or rid in ids:
            raise ValueError("invalid or duplicate opaque ID")
        ids.add(rid)
        if rec["status"] not in ("scored", "invalid"):
            raise ValueError("invalid status")
        if not isinstance(rec["metrics"], dict) or set(rec["metrics"]) != METRICS:
            raise ValueError("unexpected metrics")
        if rec["status"] == "invalid":
            if rec["flagged"] is not None or any(v is not None for v in rec["metrics"].values()):
                raise ValueError("invalid check must not export verdicts")
        elif type(rec["flagged"]) is not bool or any(type(v) is not bool for v in rec["metrics"].values()):
            raise ValueError("scored check requires boolean verdicts")
    return True


def export_batch(inputs):
    """Inputs are private evaluation-side tuples (opaque_id, report, calibration)."""
    out = {"schema_version": SCHEMA, "records": [
        export_check2(rid, report, calibration) for rid, report, calibration in inputs]}
    validate_sanitized(out["records"])
    return out
