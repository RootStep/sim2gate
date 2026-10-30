"""Synthetic checks for v0.1 public-input commitment; no sealed labels."""
import importlib.util
import json
from pathlib import Path
import unittest

root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(root))
from commit_public_inputs import commit_public_inputs


def record(rid, value):
    return {"opaque_id": rid, "measurements": {"nominal_success_rate": value}}


def raw(records):
    return json.dumps({"schema_version": "v0.1-public-features-draft", "records": records})


class TestCommitment(unittest.TestCase):
    def test_order_independent(self):
        a, b = record("opaque_A", 0.8), record("opaque_B", 0.2)
        self.assertEqual(commit_public_inputs(raw([a, b]))["sha256"],
                         commit_public_inputs(raw([b, a]))["sha256"])

    def test_changes_are_detected(self):
        self.assertNotEqual(commit_public_inputs(raw([record("opaque_A", 0.8)]))["sha256"],
                            commit_public_inputs(raw([record("opaque_A", 0.7)]))["sha256"])

    def test_forbidden_label_rejected(self):
        p = json.loads(raw([record("opaque_A", 0.8)]))
        p["records"][0]["arm"] = "known_flaw"
        with self.assertRaises(ValueError):
            commit_public_inputs(json.dumps(p))

    def test_no_identifying_data_in_commitment(self):
        c = commit_public_inputs(raw([record("opaque_A", 0.8)]))
        self.assertEqual(set(c), {"schema_version", "record_count", "sha256"})
        self.assertNotIn("opaque_A", json.dumps(c))


if __name__ == "__main__":
    unittest.main()
