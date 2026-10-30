"""Synthetic real-Check-2 adapter parity, tamper and privacy tests."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Load the unchanged production scoring source in a dependency-free CPU harness.
# The Isaac measurement module needs torch, but scoring itself needs only its
# two declared constants. This avoids installing GPU dependencies in Gate 0 CI.
import importlib.util
import types
root = Path(__file__).resolve().parents[3]
package = types.ModuleType("sim2gate")
package.__path__ = [str(root / "src" / "sim2gate")]
checks = types.ModuleType("sim2gate.checks")
checks.__path__ = [str(root / "src" / "sim2gate" / "checks")]
measure = types.ModuleType("sim2gate.checks.isaaclab_exploits")
measure.CALIBRATED_METRICS = (
    "foot_slip_p95_mps", "ground_penetration_m", "contact_force_p99_bw",
    "action_jitter_high_freq_power_share", "actuator_limit_step_share")
measure.MEASUREMENT_KEYS = (
    "dt", "contact_force_n", "foot_radius_m", "penetration_tol_m",
    "jitter_window", "flat_ground", "measurement_version", "foot_geometry")
sys.modules.update({"sim2gate": package, "sim2gate.checks": checks,
                    "sim2gate.checks.isaaclab_exploits": measure})
spec = importlib.util.spec_from_file_location(
    "sim2gate.checks.exploits_score",
    root / "src" / "sim2gate" / "checks" / "exploits_score.py")
production_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = production_module
spec.loader.exec_module(production_module)
from check2_sanitized_adapter import export_check2, export_batch, validate_sanitized, METRICS
from sim2gate.checks.exploits_score import score as production_score


class TestCheck2Adapter(unittest.TestCase):
    def test_real_production_invalid_case_parity(self):
        # Real production function invoked, intentionally incomplete synthetic
        # report must be INVALID and cannot be misrepresented as a clean pass.
        report = {"checkpoint": {"sha256": "SECRET_CHECKPOINT"},
                  "metrics": {"foot_slip_p95_mps": 999.0},
                  "recipe": "SECRET_RECIPE", "ground_truth": "SECRET_LABEL"}
        calibration = {"thresholds": {"foot_slip_p95_mps": 0.1}}
        direct = production_score(report, calibration)
        self.assertEqual(direct["status"], "invalid")
        exported = export_check2("opaque01", report, calibration)
        self.assertEqual(exported["status"], direct["status"])
        self.assertIsNone(exported["flagged"])
        self.assertTrue(all(v is None for v in exported["metrics"].values()))
        self.assertNotIn("SECRET", json.dumps(exported))
        self.assertNotIn("recipe", json.dumps(exported))

    def test_scored_verdict_parity_contract(self):
        # Fixture isolates projection parity; not evidence that a real report
        # satisfies production applicability requirements.
        original = {"status": "scored", "flagged": True,
                    "checkpoint_sha256": "SECRET", "reasons": [],
                    "metrics": {k: {"flagged": k == "foot_slip_p95_mps"}
                                for k in METRICS}}
        with patch("check2_sanitized_adapter.production_check2_score", return_value=original):
            exported = export_check2("opaque01", {"ground_truth": "SECRET"}, {})
        self.assertTrue(exported["flagged"])
        self.assertTrue(exported["metrics"]["foot_slip_p95_mps"])
        self.assertFalse(exported["metrics"]["ground_penetration_m"])
        self.assertNotIn("SECRET", json.dumps(exported))

    def test_reject_injected_identifiers(self):
        r = {"opaque_id": "opaque01", "status": "scored", "flagged": False,
             "metrics": {k: False for k in METRICS}}
        for key in ("seed", "checkpoint", "arm", "ground_truth", "recipe"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_sanitized([{**r, key: "leak"}])

    def test_no_false_clean_for_invalid(self):
        with self.assertRaises(ValueError):
            validate_sanitized([{"opaque_id": "opaque01", "status": "invalid",
                                "flagged": False, "metrics": {k: None for k in METRICS}}])

    def test_duplicate_ids_rejected(self):
        r = export_check2("opaque01", {}, {})
        with self.assertRaises(ValueError):
            validate_sanitized([r, r])

    def test_real_production_batch(self):
        result = export_batch([("opaque01", {}, {}), ("opaque02", {}, {})])
        self.assertEqual(len(result["records"]), 2)
        self.assertTrue(all(r["status"] == "invalid" for r in result["records"]))


if __name__ == "__main__":
    unittest.main()
