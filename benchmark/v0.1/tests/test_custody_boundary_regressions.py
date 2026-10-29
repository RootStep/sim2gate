"""Synthetic-only custody boundary regressions; no real held-out answer key.

Cryptographic digests detect changes relative to a trusted, externally recorded
commitment. They do not authenticate the author of a replacement commitment.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blinded_scorer_smoke import score_public_features
from custodian_verify import synthetic_unseal_with_protected_commitment, verify_prediction_commitment, verify_against_protected_digest


def run():
    return score_public_features(json.dumps({
        "schema_version": "v0.1-public-features-draft",
        "records": [
            {"opaque_id": "opaque01", "measurements": {"fall_fraction": 0.2}},
            {"opaque_id": "opaque02", "measurements": {"slip_p95_mps": 0.4}},
        ],
    }))


class CustodyBoundaryRegressions(unittest.TestCase):
    def test_prediction_invariant_under_sealed_truth_changes(self):
        result = run()
        original = json.dumps(result, sort_keys=True)
        for labels in (
            {"opaque01": True, "opaque02": False},
            {"opaque01": False, "opaque02": True},
            {"opaque01": True, "opaque02": True},
        ):
            with self.subTest(labels=labels):
                self.assertEqual(synthetic_unseal_with_protected_commitment(result, labels, result['prediction_sha256'])["total"], 2)
                self.assertEqual(json.dumps(result, sort_keys=True), original)

    def test_tamper_predictions_digest_and_ids(self):
        original = run()
        mutations = []
        a = copy.deepcopy(original)
        a["predictions"]["predictions"][0]["detect_score"] += 1
        mutations.append(a)
        b = copy.deepcopy(original)
        b["prediction_sha256"] = "0" * 64
        mutations.append(b)
        c = copy.deepcopy(original)
        c["predictions"]["predictions"][0]["opaque_id"] = "uncommitted"
        mutations.append(c)
        d = copy.deepcopy(original)
        d["predictions"]["predictions"].append(copy.deepcopy(d["predictions"]["predictions"][0]))
        mutations.append(d)
        e = copy.deepcopy(original)
        e["predictions"]["predictions"][0]["ground_truth"] = "leak"
        mutations.append(e)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    synthetic_unseal_with_protected_commitment(mutation, {"opaque01": True, "opaque02": False}, run()['prediction_sha256'])

    def test_extra_missing_and_nonboolean_key_rejected(self):
        result = run()
        for key in (
            {"opaque01": True},
            {"opaque01": True, "opaque02": False, "extra": True},
            {"opaque01": 1, "opaque02": False},
            {"opaque01": True, "opaque02": "False"},
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    synthetic_unseal_with_protected_commitment(result, key, result['prediction_sha256'])

    def test_input_metadata_injection_rejected(self):
        for injected in (
            {"ground_truth": True}, {"recipe": "hidden"}, {"seed": 3},
            {"checkpoint_sha256": "secret"}, {"key_path": "/private/key"},
        ):
            payload = {"schema_version": "v0.1-public-features-draft",
                       "records": [{"opaque_id": "opaque01",
                                    "measurements": {"fall_fraction": 0.2},
                                    **injected}]}
            with self.subTest(injected=injected):
                with self.assertRaises(ValueError):
                    score_public_features(json.dumps(payload))

    def test_protected_commitment_required_before_synthetic_unseal_with_protected_commitment(self):
        import hashlib
        original = run()
        labels = {"opaque01": True, "opaque02": False}
        protected = original["prediction_sha256"]
        self.assertEqual(synthetic_unseal_with_protected_commitment(
            original, labels, protected)["total"], 2)
        tampered = copy.deepcopy(original)
        tampered["predictions"]["predictions"][0]["detect_score"] += 2
        tampered["prediction_sha256"] = hashlib.sha256(json.dumps(
            tampered["predictions"], sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "protected pre-unseal"):
            synthetic_unseal_with_protected_commitment(tampered, labels, protected)
        with self.assertRaises(ValueError):
            synthetic_unseal_with_protected_commitment(original, labels, "0" * 64)
        with self.assertRaisesRegex(ValueError, "key IDs"):
            synthetic_unseal_with_protected_commitment(original, {"opaque01": True}, protected)

    def test_protected_external_digest_rejects_recomputed_local_digest(self):
        import hashlib
        original = run()
        protected = original["prediction_sha256"]
        self.assertEqual(verify_against_protected_digest(original, protected), protected)
        modified = copy.deepcopy(original)
        modified["predictions"]["predictions"][0]["detect_score"] += 1.0
        modified["prediction_sha256"] = hashlib.sha256(json.dumps(
            modified["predictions"], sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest()
        self.assertEqual(verify_prediction_commitment(modified), modified["prediction_sha256"])
        with self.assertRaisesRegex(ValueError, "protected pre-unseal"):
            verify_against_protected_digest(modified, protected)
        for bad in ("", "x" * 64, 123, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verify_against_protected_digest(original, bad)

    def test_digest_is_not_authentication(self):
        # A malicious actor with write access can replace both the predictions
        # and their digest. The external immutable commitment is essential.
        import hashlib
        original = run()
        replacement = copy.deepcopy(original)
        replacement["predictions"]["predictions"][0]["detect_score"] = 999.0
        body = replacement["predictions"]
        replacement["prediction_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
            .encode()).hexdigest()
        self.assertEqual(verify_prediction_commitment(replacement),
                         replacement["prediction_sha256"])
        self.assertNotEqual(original["prediction_sha256"], replacement["prediction_sha256"])


if __name__ == "__main__":
    unittest.main()
