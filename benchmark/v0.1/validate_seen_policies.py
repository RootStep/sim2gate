#!/usr/bin/env python3
"""Fail-closed seen-policy register checks against tracked historical sources.

No checkpoint bytes are accessed. Recorded hashes are not byte verification.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTER = HERE / "SEEN_POLICIES_PROVISIONAL.csv"
rows = list(csv.DictReader(REGISTER.open(newline="")))
assert len(rows) == 97, f"expected 97 seen runs, got {len(rows)}"
expected = {"v1_mvp": 23, "v2_phase1": 22, "v2_phase2": 15,
            "v2_phase2_pilot": 20, "mvp_p0v2_calibration": 3,
            "v2_phase0a": 10, "v2_phase1_pilot": 4}
assert Counter(r["cohort"] for r in rows) == expected
by_key = {}
for row in rows:
    key = (row["cohort"], row["policy_id"])
    assert key not in by_key, f"duplicate ID {key}"
    by_key[key] = row
    assert row["exclusion"] == "SEEN_EXCLUDE", key
    digest = row["checkpoint_sha256"]
    assert not digest or re.fullmatch(r"[0-9a-f]{64}", digest), key
    assert row["hash_status"] in {"RECORDED_ONLY", "BYTE_VERIFIED",
        "UNRESOLVED", "INCOMPLETE_RUN", "BYTE_VERIFIED_SHA_PREFIX_ONLY"}, key
    assert bool(digest) == (row["hash_status"] in {"RECORDED_ONLY", "BYTE_VERIFIED"}), key
    assert row["hash_source"], f"missing source attribution {key}"

def check(cohort, policy_id, digest):
    row = by_key[(cohort, policy_id)]
    assert row["checkpoint_sha256"] == digest, f"source mismatch {cohort}/{policy_id}"
    assert row["hash_status"] in ("RECORDED_ONLY", "BYTE_VERIFIED")

layer = json.loads((ROOT / "docs/v2/evidence/layer1/frozen_success_p0v5b.json").read_text())
for item in layer["sources"]:
    check("v2_phase1", item["label"], item["checkpoint_sha256"])

for relative in ("docs/v2/evidence/phase2/pilots/state.json",
                 "docs/v2/evidence/phase2/pilots_amendment1/state.json"):
    state = json.loads((ROOT / relative).read_text())
    for robot in state["robots"].values():
        check("v2_phase2_pilot", "s" + str(robot["seed"]), robot["checkpoint"]["sha256"])

cal = json.loads((ROOT / "docs/mvp-plan/calibration/p0v2/manifest.json").read_text())
for seed, data in cal["seeds"].items():
    check("mvp_p0v2_calibration", "p0v2_s" + seed, data["checkpoint"]["sha256"])

# Additional tracked MVP source hashes: ensure every registered v1 policy
# has a digest. Full alias reconciliation remains a separate review.
assert all(by_key[("v1_mvp", r["policy_id"])]["checkpoint_sha256"]
           for r in rows if r["cohort"] == "v1_mvp")
assert ("v2_phase2", "s313") in by_key

# Phase 2 study checkpoints: operator-run byte verification record (2026-10-30).
p2 = json.loads((HERE / "phase2_checkpoint_byte_verification_2026-10-30.json").read_text())["robots"]
assert len(p2) == 14 and len({r["checkpoint_sha256"] for r in p2.values()}) == 14
for policy_id, item in p2.items():
    check("v2_phase2", policy_id, item["checkpoint_sha256"])
    assert by_key[("v2_phase2", policy_id)]["hash_status"] == "BYTE_VERIFIED", policy_id
assert by_key[("v2_phase2", "s313")]["hash_status"] == "INCOMPLETE_RUN"
assert set(p2) | {"s313"} == {pid for c, pid in by_key if c == "v2_phase2"}

print("PASS: 97 seen exclusions; 15 recovered Phase 1 hashes; "
      "20 pilot and 3 calibration hashes match tracked sources; "
      "14 Phase 2 study hashes match the 2026-10-30 byte-verification record")
print("NOT VERIFIED: other cohorts' checkpoint bytes, 16 rows without any digest")
