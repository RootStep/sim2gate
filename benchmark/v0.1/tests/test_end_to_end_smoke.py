"""End-to-end synthetic leakage and commitment canaries, no real labels."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blinded_scorer_smoke import score_public_features
from custodian_verify import synthetic_unseal, verify_prediction_commitment


def payload(records):
    return json.dumps({"schema_version": "v0.1-public-features-draft", "records": records})


def rec(name, measurements):
    return {"opaque_id": name, "measurements": measurements}


class EndToEndDryRun(unittest.TestCase):
    def test_separate_scorer_and_custodian(self):
        raw = payload([rec("opaque01", {"fall_fraction": 0.3}),
                       rec("opaque02", {})])
        result = score_public_features(raw)
        self.assertEqual(verify_prediction_commitment(result), result["prediction_sha256"])
        report = synthetic_unseal(result, {"opaque01": True, "opaque02": False})
        self.assertEqual((report["scored"], report["missing"]), (1, 1))
        self.assertNotIn("True", json.dumps(result))
        self.assertNotIn("False", json.dumps(result))

    def test_planted_metadata_rejected(self):
        for name in ("ground_truth", "arm", "seed", "recipe", "checkpoint_path",
                     "reward_coefficient", "label", "key"):
            with self.subTest(name=name):
                p = json.loads(payload([rec("opaque01", {"fall_fraction": 0.3})]))
                p["records"][0][name] = "planted"
                with self.assertRaises(ValueError):
                    score_public_features(json.dumps(p))

    def test_tampering_fails_before_unseal(self):
        result = score_public_features(payload([rec("opaque01", {"fall_fraction": 0.3})]))
        result["predictions"]["predictions"][0]["detect_score"] = 0.99
        with self.assertRaises(ValueError):
            synthetic_unseal(result, {"opaque01": True})

    def test_incorrect_key_membership_rejected(self):
        result = score_public_features(payload([rec("opaque01", {"fall_fraction": 0.3})]))
        with self.assertRaises(ValueError):
            synthetic_unseal(result, {"other": True})

    def test_missing_measurements_are_not_zero_evidence(self):
        result = score_public_features(payload([rec("opaque01", {})]))
        self.assertEqual(result["predictions"]["predictions"][0]["status"], "missing_features")
        self.assertIsNone(result["predictions"]["predictions"][0]["detect_score"])


if __name__ == "__main__":
    unittest.main()
