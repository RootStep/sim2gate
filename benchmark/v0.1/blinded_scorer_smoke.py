"""Development-only blinded diagnostic scorer.

This is a deterministic synthetic smoke-test adapter, NOT the frozen Sim2Gate
method or evidence of benchmark diagnostic performance. It cannot read a label key.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_feature_boundary import parse_public_features
from commit_public_inputs import commit_public_inputs

SCORER_VERSION = "v0.1-synthetic-smoke-only"
# Illustrative fixed coefficients. Never use for scientific comparisons.
COEFFICIENTS = {
    "fall_fraction": 1.0,
    "slip_p95_mps": 0.5,
    "tracking_error_mps": 0.25,
    "actuator_limit_fraction": 0.25,
}


def score_public_features(raw):
    rows = parse_public_features(raw)
    if not rows:
        raise ValueError("empty input not allowed")
    commitment = commit_public_inputs(raw)
    predictions = []
    for opaque_id, measurements in sorted(rows):
        present = [(k, measurements[k]) for k in COEFFICIENTS if k in measurements]
        score = sum(COEFFICIENTS[k] * value for k, value in present) if present else None
        predictions.append({"opaque_id": opaque_id, "detect_score": score,
                            "status": "scored" if present else "missing_features"})
    body = {"schema_version": "v0.1-predictions-draft",
            "scorer_version": SCORER_VERSION,
            "input_sha256": commitment["sha256"],
            "predictions": predictions}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"predictions": body, "prediction_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--public-input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    result = score_public_features(Path(args.public_input).read_bytes())
    Path(args.out).write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
