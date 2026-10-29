"""Custodian-side verification, kept OUT of the blinded scorer process.

Synthetic dry-run only. Production custody requires independent private key storage,
access logs, pre-unseal timestamps and a named independent custodian.
"""
import hashlib
import json


def verify_prediction_commitment(result):
    if set(result) != {"predictions", "prediction_sha256"}:
        raise ValueError("unexpected prediction envelope")
    body = result["predictions"]
    if set(body) != {"schema_version", "scorer_version", "input_sha256", "predictions"}:
        raise ValueError("unexpected prediction fields")
    if body["schema_version"] != "v0.1-predictions-draft":
        raise ValueError("wrong schema")
    if not isinstance(body["input_sha256"], str) or len(body["input_sha256"]) != 64:
        raise ValueError("invalid input commitment")
    rows = body["predictions"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty predictions")
    ids = []
    for row in rows:
        if set(row) != {"opaque_id", "detect_score", "status"}:
            raise ValueError("unexpected prediction record")
        if not isinstance(row["opaque_id"], str) or not row["opaque_id"]:
            raise ValueError("invalid ID")
        ids.append(row["opaque_id"])
        if row["status"] not in ("scored", "missing_features"):
            raise ValueError("invalid status")
        if row["status"] == "scored":
            import math
            if type(row["detect_score"]) not in (int, float) or not math.isfinite(row["detect_score"]):
                raise ValueError("invalid score")
        elif row["detect_score"] is not None:
            raise ValueError("missing-feature score must be null")
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("duplicate or unsorted IDs")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if digest != result["prediction_sha256"]:
        raise ValueError("prediction commitment mismatch")
    return digest


def _synthetic_unseal_after_verification(result, sealed_labels):
    """Evaluate only after commitment check; never import from scorer."""
    verify_prediction_commitment(result)
    predictions = result["predictions"]["predictions"]
    ids = {r["opaque_id"] for r in predictions}
    if not isinstance(sealed_labels, dict) or set(sealed_labels) != ids:
        raise ValueError("key IDs must match predictions exactly")
    if any(type(v) is not bool for v in sealed_labels.values()):
        raise ValueError("synthetic truth must be boolean")
    scored = [r for r in predictions if r["status"] == "scored"]
    return {"prediction_sha256": result["prediction_sha256"],
            "total": len(predictions), "scored": len(scored),
            "missing": len(predictions) - len(scored),
            "synthetic_only": True}


def verify_against_protected_digest(result, protected_digest):
    """Custodian-only comparison to an independently stored pre-unseal digest.

    Caller must obtain protected_digest from a separate write-protected custody
    channel. This function cannot establish storage provenance or access control.
    """
    if not isinstance(protected_digest, str) or len(protected_digest) != 64 or any(
        c not in "0123456789abcdef" for c in protected_digest
    ):
        raise ValueError("invalid protected commitment")
    actual = verify_prediction_commitment(result)
    import hmac
    if not hmac.compare_digest(actual, protected_digest):
        raise ValueError("prediction differs from protected pre-unseal commitment")
    return actual


def synthetic_unseal_with_protected_commitment(result, sealed_labels, protected_digest):
    """Synthetic-only unseal; require trusted custodian commitment first.

    The caller must obtain protected_digest from independently protected custody
    storage, not from the scorer output. This cannot attest storage provenance.
    """
    verify_against_protected_digest(result, protected_digest)
    return _synthetic_unseal_after_verification(result, sealed_labels)


def synthetic_unseal(*args, **kwargs):
    """Legacy entry point disabled: fail closed rather than bypass protected digest."""
    raise RuntimeError(
        "legacy synthetic_unseal is disabled; use "
        "synthetic_unseal_with_protected_commitment with a custodian-protected digest"
    )
