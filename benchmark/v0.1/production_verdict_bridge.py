"""Development-only label-free bridge from real Check 1/2/5 verdicts.

Private measurement side calls the actual production scoring adapters. This
module serializes only a strict opaque ID, check status, and boolean flags.
It does not train a classifier, freeze a decision rule, or expose raw reports.
"""
import json
import re

try:
    from .check15_sanitized_adapter import export_check1, export_check5
    from .check2_sanitized_adapter import export_check2
    from .public_feature_boundary import OPAQUE_ID_PATTERN, FORBIDDEN_TERMS
except ImportError:
    from check15_sanitized_adapter import export_check1, export_check5
    from check2_sanitized_adapter import export_check2
    from public_feature_boundary import OPAQUE_ID_PATTERN, FORBIDDEN_TERMS

SCHEMA = "v0.1-production-verdicts-draft"
CHECKS = ("check1", "check2", "check5")
STATUSES = ("scored", "invalid")


def _valid_id(value):
    return bool(isinstance(value, str) and OPAQUE_ID_PATTERN.fullmatch(value)
                and not any(word in value.lower() for word in FORBIDDEN_TERMS))


def _project(record):
    if not isinstance(record, dict) or set(record) != {"opaque_id", "status", "flagged", "metrics"}:
        raise ValueError("invalid private check projection")
    if not _valid_id(record["opaque_id"]):
        raise ValueError("invalid opaque ID")
    if record["status"] not in STATUSES:
        raise ValueError("invalid check status")
    if not isinstance(record["metrics"], dict) or not record["metrics"]:
        raise ValueError("missing metric flags")
    if any(not isinstance(k, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", k)
           for k in record["metrics"]):
        raise ValueError("invalid metric name")
    values = record["metrics"].values()
    if record["status"] == "scored":
        if type(record["flagged"]) is not bool or any(type(v) is not bool for v in values):
            raise ValueError("invalid scored verdict")
    elif record["flagged"] is not None or any(v is not None for v in values):
        raise ValueError("invalid verdict must abstain")
    return {"status": record["status"], "flagged": record["flagged"],
            "metrics": dict(sorted(record["metrics"].items()))}


def export_production_verdicts(inputs):
    """inputs: iterable of private tuples (opaque_id, check1_raw, check2_raw,
    check5_physx, check5_newton, calibration1, calibration2, calibration5).

    No recipe, label, seed or checkpoint metadata is serialized.
    """
    output = []
    seen = set()
    for item in inputs:
        if not isinstance(item, (list, tuple)) or len(item) != 8:
            raise ValueError("expected eight private fields per policy")
        opaque_id, raw1, raw2, physx, newton, cal1, cal2, cal5 = item
        if not _valid_id(opaque_id) or opaque_id in seen:
            raise ValueError("invalid or duplicate opaque ID")
        seen.add(opaque_id)
        projections = {"check1": export_check1(opaque_id, raw1, cal1),
                       "check2": export_check2(opaque_id, raw2, cal2),
                       "check5": export_check5(opaque_id, physx, newton, cal5)}
        output.append({"opaque_id": opaque_id,
                       "checks": {name: _project(projections[name]) for name in CHECKS}})
    if not output:
        raise ValueError("empty input")
    return {"schema_version": SCHEMA, "records": sorted(output, key=lambda r: r["opaque_id"])}


def validate_production_verdicts(raw):
    """Reject duplicate JSON keys, unknown fields and malformed flags."""
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj
    data = json.loads(raw, object_pairs_hook=unique) if isinstance(raw, (str, bytes)) else raw
    if not isinstance(data, dict) or set(data) != {"schema_version", "records"}:
        raise ValueError("unexpected envelope fields")
    if data["schema_version"] != SCHEMA or not isinstance(data["records"], list) or not data["records"]:
        raise ValueError("invalid production verdict schema")
    seen = set()
    for record in data["records"]:
        if not isinstance(record, dict) or set(record) != {"opaque_id", "checks"}:
            raise ValueError("unexpected record fields")
        rid = record["opaque_id"]
        if not _valid_id(rid) or rid in seen:
            raise ValueError("invalid or duplicate opaque ID")
        seen.add(rid)
        if not isinstance(record["checks"], dict) or set(record["checks"]) != set(CHECKS):
            raise ValueError("missing or unexpected check")
        for name in CHECKS:
            _project({"opaque_id": rid, **record["checks"][name]})
    return True


def summarize_production_verdicts(raw):
    """Simple deterministic OR diagnostic baseline, not frozen Sim2Gate method."""
    data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    validate_production_verdicts(data)
    rows = []
    for record in sorted(data["records"], key=lambda r: r["opaque_id"]):
        checks = record["checks"]
        scored = [v["flagged"] for v in checks.values() if v["status"] == "scored"]
        rows.append({"opaque_id": record["opaque_id"],
                     "status": "scored" if len(scored) == len(CHECKS) else "missing_features",
                     "detect_score": float(any(scored)) if len(scored) == len(CHECKS) else None})
    return rows
