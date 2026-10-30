"""Label-free projections of the unchanged production Check 1 and Check 5 scores.

This module runs in the *private measurement environment*, never in the
blind scorer. Only allowlisted booleans and status leave that environment.
Not a frozen benchmark decision rule.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sim2gate.checks.robustness import score as check1_score, PARAMS
from sim2gate.checks.transfer import score as check5_score

CHECK1_NAMES = frozenset(PARAMS)
CHECK5_NAMES = frozenset(("newton_fall_free_rate", "error_ratio"))


def _verdict(opaque_id, result, names, flags):
    if not isinstance(opaque_id, str) or not opaque_id or len(opaque_id) > 80:
        raise ValueError("invalid opaque ID")
    status = result["status"]
    if status not in ("scored", "invalid"):
        raise ValueError("invalid production status")
    if set(flags) != names:
        raise ValueError("production flag set changed")
    if status == "scored":
        if type(result["flagged"]) is not bool or any(type(v) is not bool for v in flags.values()):
            raise ValueError("invalid production verdict")
        return {"opaque_id": opaque_id, "status": status, "flagged": result["flagged"],
                "metrics": dict(sorted(flags.items()))}
    return {"opaque_id": opaque_id, "status": "invalid", "flagged": None,
            "metrics": {k: None for k in sorted(names)}}


def export_check1(opaque_id, raw, calibration):
    result = check1_score(raw, calibration)
    return _verdict(opaque_id, result, CHECK1_NAMES,
                    {name: entry["flagged"] for name, entry in result["params"].items()})


def export_check5(opaque_id, physx, newton, calibration):
    result = check5_score(physx, newton, calibration)
    flagged = {name: any(r["metric"] == name for r in result["risks"])
               for name in CHECK5_NAMES}
    return _verdict(opaque_id, result, CHECK5_NAMES, flagged)
