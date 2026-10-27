#!/usr/bin/env python3
"""Validate historical exclusion CSV; no checkpoint bytes are accessed."""
import csv
import re
from collections import Counter
from pathlib import Path

path = Path(__file__).with_name("SEEN_POLICIES_PROVISIONAL.csv")
rows = list(csv.DictReader(path.open(newline="")))
assert len(rows) == 60, len(rows)
assert Counter(r["cohort"] for r in rows) == {"v1_mvp":23,"v2_phase1":22,"v2_phase2":15}
assert len({(r["cohort"],r["policy_id"]) for r in rows}) == len(rows)
for r in rows:
    assert r["exclusion"] == "SEEN_EXCLUDE", r["policy_id"]
    h = r["checkpoint_sha256"]
    assert not h or re.fullmatch("[0-9a-f]{64}", h), r["policy_id"]
    if r["hash_status"] in ("RECORDED_ONLY", "BYTE_VERIFIED"):
        assert h, r["policy_id"]
    if r["hash_status"] in ("UNRESOLVED","INCOMPLETE_RUN","BYTE_VERIFIED_SHA_PREFIX_ONLY"):
        assert not h, r["policy_id"]
assert sum(r["policy_id"] == "s313" for r in rows) == 1
print("PASS: register structure, cohort counts, exclusions and SHA syntax")
print("NOT TESTED: source bytes, scorer leakage, held-out custody")
