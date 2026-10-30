"""Synthetic regressions for F1-F5: no real keys or held-out data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blinded_scorer_smoke import score_public_features
from public_feature_boundary import parse_public_features
from custodian_verify import (verify_issued_input_binding, synthetic_unseal_once,
                              custodian_record_synthetic_commitment)


def payload():
    return json.dumps({"schema_version": "v0.1-public-features-draft",
                       "records": [{"opaque_id": "opaque01",
                                    "measurements": {"fall_fraction": 0.2}}]})


class SyntheticReviewFixes(unittest.TestCase):
    def test_duplicate_json_keys_rejected(self):
        for raw in (
            '{"schema_version":"v0.1-public-features-draft","records":[],"records":[]}',
            '{"schema_version":"v0.1-public-features-draft","records":[{"opaque_id":"opaque01","measurements":{"fall_fraction":0.2,"fall_fraction":0.4}}]}',
        ):
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                parse_public_features(raw)

    def test_ids_reject_semantic_leaks_and_punctuation(self):
        for bad in ("labeltrue", "seed1234", "recipeabc", "key12345", "opaque-01",
                    "../keyfile", "a", "opaque01/secret"):
            raw = json.loads(payload())
            raw["records"][0]["opaque_id"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_public_features(json.dumps(raw))

    def test_issued_input_and_roster_binding(self):
        result = score_public_features(payload())
        digest = result["predictions"]["input_sha256"]
        self.assertTrue(verify_issued_input_binding(result, digest, ["opaque01"]))
        with self.assertRaisesRegex(ValueError, "custodian-issued input"):
            verify_issued_input_binding(result, "0" * 64, ["opaque01"])
        with self.assertRaisesRegex(ValueError, "issued roster"):
            verify_issued_input_binding(result, digest, ["opaque02"])

    def test_write_once_commitment_and_one_shot_even_on_failure(self):
        result = score_public_features(payload())
        digest = result["predictions"]["input_sha256"]
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private"
            private.mkdir(mode=0o700)
            private.chmod(0o700)
            record = custodian_record_synthetic_commitment(result, digest, ["opaque01"], private)
            saved = json.loads(record.read_text())
            self.assertEqual(saved["prediction_sha256"], result["prediction_sha256"])
            self.assertEqual(record.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(ValueError, "already recorded"):
                custodian_record_synthetic_commitment(result, digest, ["opaque01"], private)
            with self.assertRaisesRegex(ValueError, "protected pre-unseal"):
                synthetic_unseal_once(result, {"opaque01": True}, "0" * 64,
                                      digest, ["opaque01"], private)
            with self.assertRaisesRegex(ValueError, "already attempted"):
                synthetic_unseal_once(result, {"opaque01": True}, saved["prediction_sha256"],
                                      digest, ["opaque01"], private)

    def test_one_shot_success(self):
        result = score_public_features(payload())
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "private"
            private.mkdir(mode=0o700)
            private.chmod(0o700)
            record = custodian_record_synthetic_commitment(
                result, result["predictions"]["input_sha256"], ["opaque01"], private)
            trusted = json.loads(record.read_text())
            report = synthetic_unseal_once(result, {"opaque01": True},
                trusted["prediction_sha256"], trusted["issued_input_sha256"],
                trusted["issued_ids"], private)
            self.assertEqual(report["total"], 1)


if __name__ == "__main__":
    unittest.main()
