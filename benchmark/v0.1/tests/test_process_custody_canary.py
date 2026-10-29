"""Synthetic process-separated custody canary; never handles real keys.

Scorer child receives public JSON over stdin, with an isolated working directory.
Custodian key stays in a different directory and is not passed in arguments,
environment variables or stdin. This is a *logical* separation only: a child
with the same OS identity can still read an accessible file by absolute path.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from custodian_verify import synthetic_unseal_with_protected_commitment, verify_prediction_commitment


class ProcessSeparatedCanary(unittest.TestCase):
    def test_process_handoff_and_label_invariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "custodian_private"
            public = Path(tmp) / "scorer_workspace"
            private.mkdir()
            public.mkdir()
            key_path = private / "synthetic_truth.json"
            key_path.write_text(json.dumps({"opaque01": True, "opaque02": False}))
            public_data = {"schema_version": "v0.1-public-features-draft",
                           "records": [
                               {"opaque_id": "opaque01", "measurements": {
                                   "fall_fraction": 0.2, "slip_p95_mps": 0.4,
                                   "tracking_error_mps": 0.1, "actuator_limit_fraction": 0.1}},
                               {"opaque_id": "opaque02", "measurements": {
                                   "fall_fraction": 0.0, "slip_p95_mps": 0.1,
                                   "tracking_error_mps": 0.1, "actuator_limit_fraction": 0.1}}]}
            script = ROOT / "blinded_scorer_smoke.py"
            input_path = public / "features.json"
            output_path = public / "predictions.json"
            input_path.write_text(json.dumps(public_data))
            # Child has no private key path in args, environment or input.
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
            cmd = [sys.executable, str(script), "--public-input", str(input_path), "--out", str(output_path)]
            first = subprocess.run(cmd, text=True,
                                   capture_output=True, cwd=public, env=env, check=True)
            first_predictions = output_path.read_text()
            output = json.loads(first_predictions)
            verify_prediction_commitment(output)
            key = json.loads(key_path.read_text())
            self.assertEqual(synthetic_unseal_with_protected_commitment(output, key, output['prediction_sha256'])["total"], 2)
            key_path.write_text(json.dumps({"opaque01": False, "opaque02": True}))
            second = subprocess.run(cmd, input=json.dumps(public_data), text=True,
                                    capture_output=True, cwd=public, env=env, check=True)
            self.assertEqual(first_predictions, output_path.read_text(),
                             "predictions changed when only sealed synthetic truth changed")
            self.assertNotIn(str(key_path), first_predictions)
            self.assertNotIn("custodian_private", first_predictions)


if __name__ == "__main__":
    unittest.main()
