"""Regression tests for the seen-policy exclusion register (no checkpoint bytes accessed)."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_seen_policies as v


def rows():
    return v.load()


class SeenRegister(unittest.TestCase):
    def test_current_register_passes(self):
        statuses, n_tracked = v.validate(rows())
        self.assertEqual(n_tracked, 66)
        self.assertEqual(statuses["BYTE_VERIFIED"], 5)

    def expect_fail(self, mutate, message):
        r = copy.deepcopy(rows())
        mutate(r)
        with self.assertRaisesRegex(ValueError, message):
            v.validate(r)

    def test_dropping_a_tracked_digest_fails(self):
        def drop(r):
            row = next(x for x in r if x["policy_id"] == "p2pilot_R-slip_k0_1_s201")
            row["checkpoint_sha256"], row["hash_status"] = "", "UNRESOLVED"
        self.expect_fail(drop, "missing from")

    def test_misattributed_digest_fails(self):
        def swap(r):
            a = next(x for x in r if x["policy_id"] == "p0v5b_s107")
            b = next(x for x in r if x["policy_id"] == "p0v5b_s108")
            a["checkpoint_sha256"], b["checkpoint_sha256"] = b["checkpoint_sha256"], a["checkpoint_sha256"]
        self.expect_fail(swap, "misattributed")

    def test_duplicate_digest_fails(self):
        def dup(r):
            r[1]["checkpoint_sha256"] = r[0]["checkpoint_sha256"]
        self.expect_fail(dup, "also registered")

    def test_exclusion_cannot_change(self):
        def free(r):
            r[0]["exclusion"] = "ELIGIBLE"
        self.expect_fail(free, "SEEN_EXCLUDE")

    def test_recorded_only_requires_source(self):
        def nosrc(r):
            next(x for x in r if x["policy_id"] == "p0v5b_s107")["hash_source"] = ""
        self.expect_fail(nosrc, "requires hash_source")

    def test_unresolved_row_cannot_carry_guessed_digest(self):
        def guess(r):
            next(x for x in r if x["policy_id"] == "s306")["checkpoint_sha256"] = "a" * 64
        self.expect_fail(guess, "must not carry a digest")

    def test_alias_must_be_recorded(self):
        def unalias(r):
            next(x for x in r if x["policy_id"] == "p0v2_s1")["aliases"] = ""
        self.expect_fail(unalias, "alias 'BSUBF'")


if __name__ == "__main__":
    unittest.main()
