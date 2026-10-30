"""Blinded v0.1 prediction commitment envelope (development-only scaffolding).

Only opaque identifiers and numeric diagnostic measurements cross this boundary.
No intervention labels, recipe identifiers, seeds, paths or checkpoint metadata.
This module does not perform Sim2Gate inference or evaluate accuracy.
"""
import hashlib
import json
from pathlib import Path

try:
    from .public_feature_boundary import parse_public_features
except ImportError:
    from public_feature_boundary import parse_public_features


def commit_public_inputs(raw):
    """Validate the scorer boundary and return a deterministic hash commitment."""
    rows = parse_public_features(raw)
    canonical = json.dumps(
        [{"opaque_id": rid, "measurements": dict(sorted(features.items()))}
         for rid, features in sorted(rows)],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return {"schema_version": "v0.1-input-commitment-draft",
            "record_count": len(rows),
            "sha256": hashlib.sha256(canonical).hexdigest()}


def write_commitment(raw, output_path):
    """Write only commitment metadata; never copy source or sealed keys."""
    result = commit_public_inputs(raw)
    Path(output_path).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result
