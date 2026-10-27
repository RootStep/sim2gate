"""Strict input boundary for a future blinded v0.1 benchmark scorer.

This module does not implement Sim2Gate scoring or authorize benchmark training.
"""
import json

ALLOWED_TOP_LEVEL = frozenset(("schema_version", "records"))
ALLOWED_RECORD = frozenset(("opaque_id", "measurements"))
ALLOWED_MEASUREMENTS = frozenset((
    "nominal_success_rate",
    "tracking_error_mps",
    "fall_fraction",
    "slip_p95_mps",
    "actuator_limit_fraction",
    "latency_cliff_index",
    "newton_fall_free_fraction",
    "newton_tracking_error_ratio",
))
FORBIDDEN_TERMS = frozenset((
    "seed", "arm", "label", "cause", "recipe", "reward", "checkpoint",
    "path", "filename", "intervention", "ground_truth", "key",
))


def parse_public_features(raw):
    """Reject unexpected metadata rather than silently dropping it."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != ALLOWED_TOP_LEVEL:
        raise ValueError("unexpected top-level input fields")
    if payload["schema_version"] != "v0.1-public-features-draft":
        raise ValueError("unsupported schema")
    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    seen = set()
    output = []
    for rec in records:
        if not isinstance(rec, dict) or set(rec) != ALLOWED_RECORD:
            raise ValueError("unexpected record fields")
        opaque_id = rec["opaque_id"]
        if not isinstance(opaque_id, str) or not opaque_id or len(opaque_id) > 80:
            raise ValueError("invalid opaque ID")
        if opaque_id in seen:
            raise ValueError("duplicate opaque ID")
        seen.add(opaque_id)
        measurements = rec["measurements"]
        if not isinstance(measurements, dict) or not set(measurements).issubset(ALLOWED_MEASUREMENTS):
            raise ValueError("unknown measurement field")
        clean = {}
        for name, value in measurements.items():
            if type(value) not in (int, float):
                raise ValueError("measurement must be numeric")
            import math
            if not math.isfinite(value):
                raise ValueError("measurement must be finite")
            clean[name] = float(value)
        output.append((opaque_id, clean))
    return output
