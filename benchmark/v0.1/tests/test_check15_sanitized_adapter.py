"""Projection parity tests for real Check 1/5 scoring, without Isaac Sim."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check15_sanitized_adapter import export_check1, export_check5, CHECK1_NAMES, CHECK5_NAMES


class TestCheck15Adapter(unittest.TestCase):
    def test_check1_projection(self):
        result = {"status": "scored", "flagged": True, "checkpoint_sha256": "SECRET",
                  "params": {k: {"flagged": k == "ground_friction", "training_range": "SECRET"}
                             for k in CHECK1_NAMES}}
        with patch("check15_sanitized_adapter.check1_score", return_value=result) as fn:
            exported = export_check1("opaque01", {"ground_truth": "SECRET"}, {})
            fn.assert_called_once()
        self.assertTrue(exported["metrics"]["ground_friction"])
        self.assertNotIn("SECRET", repr(exported))

    def test_check5_projection(self):
        result = {"status": "scored", "flagged": True, "checkpoint_sha256": "SECRET",
                  "risks": [{"metric": "error_ratio"}]}
        with patch("check15_sanitized_adapter.check5_score", return_value=result) as fn:
            exported = export_check5("opaque02", {}, {}, {})
            fn.assert_called_once()
        self.assertTrue(exported["metrics"]["error_ratio"])
        self.assertFalse(exported["metrics"]["newton_fall_free_rate"])
        self.assertNotIn("SECRET", repr(exported))

    def test_invalid_never_clean(self):
        r1 = {"status": "invalid", "flagged": None,
              "params": {k: {"flagged": False} for k in CHECK1_NAMES}}
        r5 = {"status": "invalid", "flagged": None, "risks": []}
        with patch("check15_sanitized_adapter.check1_score", return_value=r1):
            self.assertIsNone(export_check1("opaque01", {}, {})["flagged"])
        with patch("check15_sanitized_adapter.check5_score", return_value=r5):
            self.assertTrue(all(v is None for v in export_check5("opaque01", {}, {}, {})["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
