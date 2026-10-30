"""Synthetic adversarial tests for the draft public-feature boundary."""
import json
import unittest
from pathlib import Path
import importlib.util

path = Path(__file__).resolve().parents[1] / "public_feature_boundary.py"
spec = importlib.util.spec_from_file_location("public_feature_boundary", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def payload():
    return {"schema_version": "v0.1-public-features-draft",
            "records": [{"opaque_id": "RANDOM_A", "measurements": {"nominal_success_rate": 0.91}}]}


class TestPublicBoundary(unittest.TestCase):
    def parse(self, obj):
        return mod.parse_public_features(json.dumps(obj))

    def test_accept_public_measurement(self):
        self.assertEqual(self.parse(payload()), [("RANDOM_A", {"nominal_success_rate": 0.91})])

    def test_reject_label_at_top_level(self):
        p = payload()
        p["ground_truth"] = "flawed"
        with self.assertRaises(ValueError):
            self.parse(p)

    def test_reject_recipe_at_record_level(self):
        p = payload()
        p["records"][0]["recipe"] = "D1"
        with self.assertRaises(ValueError):
            self.parse(p)

    def test_reject_reward_coefficient(self):
        p = payload()
        p["records"][0]["measurements"]["reward_coefficient"] = 2.0
        with self.assertRaises(ValueError):
            self.parse(p)

    def test_reject_checkpoint_path(self):
        p = payload()
        p["records"][0]["checkpoint_path"] = "/secret/flaw.pt"
        with self.assertRaises(ValueError):
            self.parse(p)

    def test_reject_duplicate_id(self):
        p = payload()
        p["records"].append(p["records"][0].copy())
        with self.assertRaises(ValueError):
            self.parse(p)

    def test_reject_nonfinite(self):
        p = payload()
        p["records"][0]["measurements"]["nominal_success_rate"] = float("nan")
        with self.assertRaises(ValueError):
            self.parse(p)

    def test_reject_key_in_measurements(self):
        p = payload()
        p["records"][0]["measurements"]["key"] = 1
        with self.assertRaises(ValueError):
            self.parse(p)


if __name__ == "__main__":
    unittest.main()
