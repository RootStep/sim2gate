"""Linux-only synthetic OS credential isolation test.

Creates a fake answer key automatically. Runs the scorer as the unprivileged
'nobody' account, with a private directory that the scorer cannot traverse.
This tests actual OS discretionary access controls, not merely separate paths.
No real keys or held-out labels are used.
"""
import json
import os
import shutil
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from custodian_verify import synthetic_unseal_with_protected_commitment, verify_prediction_commitment, verify_against_protected_digest


@unittest.skipUnless(sys.platform.startswith("linux") and os.geteuid() == 0,
                     "requires a Linux root-controlled test runner to drop scorer privileges")
class SyntheticOsIsolation(unittest.TestCase):
    def test_scoring_identity_cannot_read_synthetic_truth(self):
        nobody = pwd.getpwnam("nobody")
        self.assertNotEqual(nobody.pw_uid, 0)
        with tempfile.TemporaryDirectory(prefix="rootstep-custody-") as temp:
            base = Path(temp)
            base.chmod(0o755)
            private = base / "custodian_private"
            public = base / "scorer_public"
            private.mkdir(mode=0o700)
            public.mkdir(mode=0o755)
            private.chmod(0o700)
            public.chmod(0o755)
            key = private / "synthetic_answer_key.json"
            key.write_text(json.dumps({"opaque01": True, "opaque02": False}))
            key.chmod(0o600)
            self.assertEqual(json.loads(key.read_text())["opaque01"], True)
            commitment = private / "protected_prediction_sha256.txt"
            input_path = public / "public_features.json"
            output_path = public / "predictions.json"
            input_path.write_text(json.dumps({
                "schema_version": "v0.1-public-features-draft",
                "records": [
                    {"opaque_id": "opaque01", "measurements": {"fall_fraction": 0.2}},
                    {"opaque_id": "opaque02", "measurements": {"fall_fraction": 0.0}},
                ],
            }))
            input_path.chmod(0o644)
            os.chown(public, nobody.pw_uid, nobody.pw_gid)
            public.chmod(0o700)  # scorer-owned private output directory
            # A child script attempts the private path *explicitly* and
            # executes the scorer only after access is denied.
            # Copy only allowlisted scorer modules into the accessible sandbox. 
            # The runner checkout is intentionally inaccessible to the nobody account. 
            for filename in ("blinded_scorer_smoke.py", "public_feature_boundary.py", 
                             "commit_public_inputs.py"): 
                shutil.copyfile(ROOT / filename, public / filename) 
                (public / filename).chmod(0o644) 
            child = public / "scorer_child.py"
            child.write_text(
                "import pathlib, runpy, sys\n"
                "private = pathlib.Path(sys.argv[1])\n"
                "try:\n"
                "    private.read_bytes()\n"
                "except PermissionError:\n"
                "    pass\n"
                "else:\n"
                "    raise RuntimeError('synthetic answer key leaked to scorer')\n"
                "try:\n"
                "    pathlib.Path(sys.argv[5]).read_bytes()\n"
                "except PermissionError:\n"
                "    pass\n"
                "else:\n"
                "    raise RuntimeError('protected digest leaked to scorer')\n"
                "sys.argv = [sys.argv[2], '--public-input', sys.argv[3], '--out', sys.argv[4]]\n"
                "runpy.run_path(sys.argv[0], run_name='__main__')\n"
            )
            child.chmod(0o644)

            def drop_privileges():
                os.setgroups([])
                os.setgid(nobody.pw_gid)
                os.setuid(nobody.pw_uid)

            env = {"PATH": os.environ.get("PATH", ""),
                   "PYTHONIOENCODING": "utf-8"}
            cmd = [sys.executable, str(child), str(key),
                   str(public / "blinded_scorer_smoke.py"),
                   str(input_path), str(output_path), str(commitment)]
            proc = subprocess.run(cmd, cwd=public, env=env,
                                  preexec_fn=drop_privileges,
                                  text=True, capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(output_path.read_text())
            commitment.write_text(result["prediction_sha256"])
            commitment.chmod(0o600)
            self.assertEqual(commitment.stat().st_mode & 0o777, 0o600)
            self.assertEqual(verify_against_protected_digest(
                result, commitment.read_text()), result["prediction_sha256"])
            # A scorer with the exact protected path must still receive permission denied.
            access_probe = subprocess.run(
                [sys.executable, "-c",
                 "import pathlib,sys; pathlib.Path(sys.argv[1]).read_bytes()",
                 str(commitment)], cwd=public, env=env,
                preexec_fn=drop_privileges, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(access_probe.returncode, 0)
            self.assertIn("PermissionError", access_probe.stderr)
            # The scorer also must not replace the custodian-owned commitment.
            write_probe = subprocess.run(
                [sys.executable, "-c",
                 "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('0'*64)",
                 str(commitment)], cwd=public, env=env,
                preexec_fn=drop_privileges, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(write_probe.returncode, 0)
            self.assertIn("PermissionError", write_probe.stderr)
            self.assertEqual(commitment.read_text(), result["prediction_sha256"])
            # Cannot unlink/replace via the inaccessible parent directory either.
            unlink_probe = subprocess.run(
                [sys.executable, "-c",
                 "import pathlib,sys; pathlib.Path(sys.argv[1]).unlink()",
                 str(commitment)], cwd=public, env=env,
                preexec_fn=drop_privileges, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(unlink_probe.returncode, 0)
            self.assertIn("PermissionError", unlink_probe.stderr)
            self.assertTrue(commitment.is_file())
            import copy
            import hashlib
            modified = copy.deepcopy(result)
            modified["predictions"]["predictions"][0]["detect_score"] += 1
            modified["prediction_sha256"] = hashlib.sha256(json.dumps(
                modified["predictions"], sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode()).hexdigest()
            self.assertEqual(verify_prediction_commitment(modified),
                             modified["prediction_sha256"])
            with self.assertRaisesRegex(ValueError, "protected pre-unseal"):
                verify_against_protected_digest(modified, commitment.read_text())
            self.assertEqual(verify_prediction_commitment(result),
                             result["prediction_sha256"])
            self.assertEqual(synthetic_unseal_with_protected_commitment(
                result, json.loads(key.read_text()), commitment.read_text())["total"], 2)
            self.assertNotIn("synthetic_answer_key", json.dumps(result))
            self.assertNotIn("custodian_private", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
