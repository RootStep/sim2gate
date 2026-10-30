#!/usr/bin/env python3
"""Validate the historical seen-policy exclusion register against every tracked digest source.

No checkpoint bytes are accessed. A digest found in a tracked source but missing from the
register (or attached to a different policy) is a failure: the register must exclude every
policy the repository already knows was trained.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTER = HERE / "SEEN_POLICIES_PROVISIONAL.csv"

COLUMNS = ["cohort", "policy_id", "seed", "arm", "checkpoint_sha256", "hash_status", "qualification",
           "exclusion", "source_note", "hash_source", "aliases", "notes"]
COHORT_COUNTS = {"v1_mvp": 23, "v2_phase1": 22, "v2_phase2": 15, "v2_phase2_pilot": 20,
                 "mvp_p0v2_calibration": 3, "v2_phase0a": 10, "v2_phase1_pilot": 4}
WITH_DIGEST = {"RECORDED_ONLY", "BYTE_VERIFIED"}
WITHOUT_DIGEST = {"UNRESOLVED", "INCOMPLETE_RUN", "BYTE_VERIFIED_SHA_PREFIX_ONLY"}
SHA = re.compile(r"[0-9a-f]{64}")


def load(path=REGISTER):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def tracked_digests(root=ROOT):
    """Return {digest: expected policy_id} from every tracked source that records checkpoint digests."""
    root = Path(root)
    found = {}

    def add(digest, policy_id, source):
        if not SHA.fullmatch(digest):
            raise ValueError(f"{source}: malformed digest for {policy_id}")
        if digest in found and found[digest] != policy_id:
            raise ValueError(f"{source}: digest {digest[:12]} maps to {policy_id} and {found[digest]}")
        found[digest] = policy_id

    step4 = json.loads((root / "docs/mvp-plan/evidence/step4/key.json").read_text())["policies"]
    for pid, v in step4.items():
        add(v["sha256"], pid, "step4/key.json")
    layer1 = json.loads((root / "docs/v2/evidence/layer1/frozen_success_p0v5b.json").read_text())["sources"]
    check2 = json.loads((root / "docs/v2/evidence/layer1/check2_limit_v2.json").read_text())["sources"]
    if [(s["label"], s["checkpoint_sha256"]) for s in layer1] != [(Path(s["report"]).name, s["checkpoint_sha256"]) for s in check2]:
        raise ValueError("Layer 1 sources disagree")
    for s in layer1:
        add(s["checkpoint_sha256"], s["label"], "layer1")
    for f in ("docs/v2/evidence/phase2/pilots/state.json", "docs/v2/evidence/phase2/pilots_amendment1/state.json"):
        doc = json.loads((root / f).read_text())
        for name, v in doc["robots"].items():
            add(v["checkpoint"]["sha256"], name, f)
        for name, v in doc.get("controls", {}).items():
            add(v["checkpoint_sha256"], name, f)
    for sd, s in json.loads((root / "docs/mvp-plan/calibration/p0v2/manifest.json").read_text())["seeds"].items():
        add(s["checkpoint"]["sha256"], f"p0v2_s{sd}", "calibration/p0v2")
    # Alias-only sources: their digests must already be registered under the canonical policy.
    aliases = {}
    for f in ("docs/mvp-plan/calibration/p0v4/manifest.json",):
        for sd, s in json.loads((root / f).read_text())["seeds"].items():
            aliases.setdefault(s["checkpoint"]["sha256"], []).append(f"p0v4 calibration seed {sd}")
    for pid, v in json.loads((root / "docs/mvp-plan/evidence/pre-freeze/key.json").read_text())["policies"].items():
        aliases.setdefault(v["sha256"], []).append(pid)
    return found, aliases


def validate(rows, root=ROOT):
    root = Path(root)
    if not rows or list(rows[0].keys()) != COLUMNS:
        raise ValueError("register columns changed")
    if Counter(r["cohort"] for r in rows) != COHORT_COUNTS:
        raise ValueError(f"cohort counts changed: {dict(Counter(r['cohort'] for r in rows))}")
    if len({(r["cohort"], r["policy_id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate cohort/policy_id")
    by_digest = {}
    for r in rows:
        pid, h, st = r["policy_id"], r["checkpoint_sha256"], r["hash_status"]
        if r["exclusion"] != "SEEN_EXCLUDE":
            raise ValueError(f"{pid}: exclusion must remain SEEN_EXCLUDE")
        if st in WITH_DIGEST:
            if not SHA.fullmatch(h):
                raise ValueError(f"{pid}: {st} requires a full lowercase SHA-256")
            if not r["hash_source"]:
                raise ValueError(f"{pid}: {st} requires hash_source")
        elif st in WITHOUT_DIGEST:
            if h:
                raise ValueError(f"{pid}: {st} must not carry a digest")
        else:
            raise ValueError(f"{pid}: unknown hash_status {st!r}")
        for part in r["hash_source"].split(";"):
            part = part.strip()
            if part.startswith(("docs/", "benchmark/", "src/")) and not (root / part).is_file():
                raise ValueError(f"{pid}: hash_source path missing: {part}")
        if h:
            if h in by_digest:
                raise ValueError(f"{pid}: digest also registered for {by_digest[h]}")
            by_digest[h] = pid
    if sum(r["policy_id"] == "s313" and r["hash_status"] == "INCOMPLETE_RUN" for r in rows) != 1:
        raise ValueError("s313 must remain a single INCOMPLETE_RUN row")
    found, aliases = tracked_digests(root)
    for h, pid in found.items():
        if by_digest.get(h) != pid:
            raise ValueError(f"tracked digest {h[:12]} for {pid} is {'missing from' if h not in by_digest else 'misattributed in'} the register")
    alias_text = {r["policy_id"]: r["aliases"] for r in rows}
    for h, names in aliases.items():
        if h not in by_digest:
            raise ValueError(f"alias-only digest {h[:12]} ({names}) is not registered")
        for name in names:
            if name not in alias_text[by_digest[h]]:
                raise ValueError(f"{by_digest[h]}: alias {name!r} not recorded")
    return Counter(r["hash_status"] for r in rows), len(found)


def main():
    rows = load()
    statuses, n_tracked = validate(rows)
    print(f"PASS: {len(rows)} rows, all SEEN_EXCLUDE; statuses {dict(sorted(statuses.items()))}")
    print(f"PASS: all {n_tracked} digests in tracked repository sources are registered under the expected policy; aliases recorded")
    print("NOT TESTED: checkpoint bytes, external archives, operator ledgers, scorer leakage, held-out custody")


if __name__ == "__main__":
    main()
