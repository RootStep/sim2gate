"""Register validator must reject missing and conflicting known checkpoint digests."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "benchmark/v0.1/validate_seen_policies.py"
REGISTER = ROOT / "benchmark/v0.1/SEEN_POLICIES_PROVISIONAL.csv"

class RegisterV2Tests(unittest.TestCase):
    def test_tracked_sources_match(self):
        subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True, capture_output=True)

    def test_source_hash_missing_or_conflicting_fails(self):
        original = REGISTER.read_bytes()
        try:
            for replacement in ("", "0" * 64):
                with self.subTest(replacement=replacement):
                    rows = list(csv.DictReader(original.decode().splitlines()))
                    row = next(r for r in rows if r["policy_id"] == "p0v5b_s107")
                    row["checkpoint_sha256"] = replacement
                    if not replacement:
                        row["hash_status"] = "UNRESOLVED"
                    with REGISTER.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                    p = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                                       capture_output=True, text=True)
                    self.assertNotEqual(p.returncode, 0, p.stdout)
        finally:
            REGISTER.write_bytes(original)

    def test_phase2_byte_verified_digest_tampering_fails(self):
        original = REGISTER.read_bytes()
        try:
            for cohort, policy_id, field, value in (("v2_phase2", "s306", "checkpoint_sha256", "0" * 64),
                                                    ("v2_phase2", "s306", "hash_status", "RECORDED_ONLY"),
                                                    ("v2_phase2", "s301", "checkpoint_sha256", "f" * 64),
                                                    ("v2_phase1", "p0v5b_s107", "hash_status", "RECORDED_ONLY")):
                with self.subTest(policy_id=policy_id, field=field):
                    rows = list(csv.DictReader(original.decode().splitlines()))
                    row = next(r for r in rows if r["cohort"] == cohort and r["policy_id"] == policy_id)
                    row[field] = value
                    with REGISTER.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                    p = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT,
                                       capture_output=True, text=True)
                    self.assertNotEqual(p.returncode, 0, p.stdout)
        finally:
            REGISTER.write_bytes(original)

if __name__ == "__main__":
    unittest.main()
