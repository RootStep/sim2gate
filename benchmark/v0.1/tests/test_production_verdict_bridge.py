"""Synthetic production bridge contract: no held-out policies or labels."""
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/v0.1"))
sys.path.insert(0, str(ROOT / "src"))
package = types.ModuleType("sim2gate")
package.__path__ = [str(ROOT / "src/sim2gate")]
checks = types.ModuleType("sim2gate.checks")
checks.__path__ = [str(ROOT / "src/sim2gate/checks")]
measurement = types.ModuleType("sim2gate.checks.isaaclab_exploits")
measurement.CALIBRATED_METRICS = (
    "foot_slip_p95_mps", "ground_penetration_m", "contact_force_p99_bw",
    "action_jitter_high_freq_power_share", "actuator_limit_step_share")
measurement.MEASUREMENT_KEYS = (
    "dt", "contact_force_n", "foot_radius_m", "penetration_tol_m",
    "jitter_window", "flat_ground", "measurement_version", "foot_geometry")
sys.modules.update({"sim2gate": package, "sim2gate.checks": checks,
                    "sim2gate.checks.isaaclab_exploits": measurement})
from production_verdict_bridge import validate_production_verdicts, summarize_production_verdicts, _project

def valid():
    return {"schema_version": "v0.1-production-verdicts-draft",
            "records": [{"opaque_id": "opaque01", "checks": {
                name: {"status": "scored", "flagged": False, "metrics": {"metric_a": False}}
                for name in ("check1", "check2", "check5")}}]}

class BridgeContract(unittest.TestCase):
    def test_valid_or_baseline(self):
        payload = valid()
        self.assertTrue(validate_production_verdicts(json.dumps(payload)))
        self.assertEqual(summarize_production_verdicts(payload),
                         [{"opaque_id": "opaque01", "status": "scored", "detect_score": 0.0}])
        payload["records"][0]["checks"]["check2"]["flagged"] = True
        self.assertEqual(summarize_production_verdicts(payload)[0]["detect_score"], 1.0)

    def test_invalid_abstains(self):
        payload = valid()
        payload["records"][0]["checks"]["check5"] = {
            "status": "invalid", "flagged": None, "metrics": {"metric_a": None}}
        self.assertEqual(summarize_production_verdicts(payload)[0]["status"], "missing_features")
        self.assertIsNone(summarize_production_verdicts(payload)[0]["detect_score"])

    def test_rejects_extra_fields_and_duplicate_keys(self):
        payload = valid()
        payload["records"][0]["recipe"] = "PRIVATE_LABEL"
        with self.assertRaises(ValueError):
            validate_production_verdicts(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            validate_production_verdicts('{"schema_version":"v0.1-production-verdicts-draft","records":[],"records":[]}')

    def test_rejects_label_like_ids_and_bad_flags(self):
        for rid in ("seed12345", "label12345", "opaque-01"):
            payload = valid()
            payload["records"][0]["opaque_id"] = rid
            with self.subTest(rid=rid), self.assertRaises(ValueError):
                validate_production_verdicts(payload)
        for flag in (1, "false", None):
            payload = valid()
            payload["records"][0]["checks"]["check1"]["flagged"] = flag
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                validate_production_verdicts(payload)

    def test_private_projection_rejects_extra_metadata(self):
        with self.assertRaises(ValueError):
            _project({"opaque_id": "opaque01", "status": "scored", "flagged": True,
                      "metrics": {"metric_a": True}, "checkpoint_sha256": "SECRET"})

if __name__ == "__main__":
    unittest.main()
