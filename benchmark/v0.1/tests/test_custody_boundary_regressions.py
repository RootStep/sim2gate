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
from custodian_verify import synthetic_unseal, verify_prediction_commitment


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
                self.assertEqual(synthetic_unseal(result, labels)["total"], 2)
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
                    synthetic_unseal(mutation, {"opaque01": True, "opaque02": False})

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
                    synthetic_unseal(result, key)

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
