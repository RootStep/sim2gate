"""Synthetic tests for the read-only checkpoint verifier. No real checkpoints or archives are used."""
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_checkpoints as vc

REGISTER = Path(vc.__file__).resolve().parent / "SEEN_POLICIES_PROVISIONAL.csv"


def sha(b):
    return hashlib.sha256(b).hexdigest()


class VerifyCheckpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.known = os.urandom(4096)        # stands in for a registered checkpoint
        self.unknown = os.urandom(1024)      # stands in for an unregistered checkpoint
        run = self.d / "run" / "robots"
        (run / "p0v5b_s107" / "checkpoint").mkdir(parents=True)
        (run / "extra" / "checkpoint").mkdir(parents=True)
        (run / "p0v5b_s107" / "checkpoint" / "model_1999.pt").write_bytes(self.known)
        (run / "extra" / "checkpoint" / "model_1999.pt").write_bytes(self.unknown)
        (run / "notes.json").write_text("{}")
        self.archive = self.d / "synthetic.tgz"
        with tarfile.open(self.archive, "w:gz") as t:
            t.add(self.d / "run", arcname="run")
        # Temporary register: real columns, one row's digest replaced by the synthetic file's digest.
        with REGISTER.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.fields = list(rows[0].keys())
        rows[0]["checkpoint_sha256"] = sha(self.known)
        self.policy = rows[0]["policy_id"]
        self.register = self.d / "register.csv"
        with self.register.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            w.writerows(rows)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *paths, json_out=None):
        args = [str(p) for p in paths] + ["--register", str(self.register)]
        if json_out:
            args += ["--json", str(json_out)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            vc.main(args)
        return out.getvalue()

    def test_archive_digest_matches_independent_hash(self):
        files, archive = vc.scan(self.archive)
        self.assertEqual(archive["sha256"], sha(self.archive.read_bytes()))
        self.assertEqual(archive["bytes"], self.archive.stat().st_size)
        self.assertEqual(archive["recorded_as"], "NO RECORDED DIGEST MATCH")
        self.assertEqual(sorted(d for _, d, _ in files), sorted([sha(self.known), sha(self.unknown)]))

    def test_known_bundle_digest_is_recognised(self):
        digest = sha(self.archive.read_bytes())
        original = dict(vc.KNOWN_BUNDLES)
        try:
            vc.KNOWN_BUNDLES[digest] = "synthetic.tgz (test)"
            self.assertEqual(vc.scan(self.archive)[1]["recorded_as"], "synthetic.tgz (test)")
        finally:
            vc.KNOWN_BUNDLES.clear()
            vc.KNOWN_BUNDLES.update(original)

    def test_match_and_unregistered_reported(self):
        out = self.run_tool(self.archive)
        self.assertIn(f"MATCH   {self.policy}", out)
        self.assertIn(sha(self.known), out)
        self.assertIn(f"UNREGISTERED {sha(self.unknown)}", out)
        self.assertIn("1 register rows matched by bytes", out)

    def test_directory_and_single_file_inputs(self):
        files, archive = vc.scan(self.d / "run")
        self.assertIsNone(archive)
        self.assertEqual(len(files), 2)
        single = self.d / "run" / "robots" / "extra" / "checkpoint" / "model_1999.pt"
        self.assertEqual(vc.scan(single)[0][0][1], sha(self.unknown))

    def test_non_checkpoint_members_ignored(self):
        files, _ = vc.scan(self.archive)
        self.assertTrue(all(loc.endswith(".pt") for loc, _, _ in files))

    def test_register_never_modified(self):
        before = sha(self.register.read_bytes())
        real_before = sha(REGISTER.read_bytes())
        self.run_tool(self.archive, self.d / "run", json_out=self.d / "evidence.json")
        self.assertEqual(sha(self.register.read_bytes()), before)
        self.assertEqual(sha(REGISTER.read_bytes()), real_before)

    def test_archive_not_extracted_to_disk(self):
        before = sorted(p.name for p in self.d.iterdir())
        self.run_tool(self.archive)
        self.assertEqual(sorted(p.name for p in self.d.iterdir()), before)

    def test_evidence_record_contents(self):
        ev = self.d / "evidence.json"
        self.run_tool(self.archive, json_out=ev)
        rec = json.loads(ev.read_text())
        self.assertEqual(rec["register_sha256"], sha(self.register.read_bytes()))
        self.assertEqual(rec["tool_sha256"], sha(Path(vc.__file__).read_bytes()))
        self.assertEqual(rec["matches"][self.policy]["sha256"], sha(self.known))
        self.assertEqual([u["sha256"] for u in rec["unregistered"]], [sha(self.unknown)])
        self.assertEqual(rec["archives"][0]["sha256"], sha(self.archive.read_bytes()))

    def test_missing_path_fails(self):
        with self.assertRaises(SystemExit):
            vc.scan(self.d / "does_not_exist.tgz")


if __name__ == "__main__":
    unittest.main()
