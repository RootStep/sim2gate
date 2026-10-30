#!/usr/bin/env python3
"""Read-only checkpoint recovery check against the seen-policy register.

Usage:
    python verify_checkpoints.py PATH [PATH ...] [--register CSV] [--json OUT]

PATH may be a .tgz/.tar archive, a directory, or a single .pt file. Archives are streamed in one
pass: the archive's own SHA-256 and every *.pt member's SHA-256 are computed without extracting
anything to disk. Nothing is written except the optional --json evidence record.

Results:
  MATCH       a register digest was found in these bytes (byte-level evidence for that row)
  UNREGISTERED a .pt digest not in the register (candidate for an UNRESOLVED row; review by hand,
              never copy into the register without confirming run identity from the archive's own
              records)
Register rows whose digests were not seen are listed as NOT_SEEN. The register is never modified.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNK = 1 << 20

# Recorded bundle digests, copied from tracked evidence READMEs (source in the comment).
KNOWN_BUNDLES = {
    "36bb9b75caf2ab7d37482bf7dc79c5ddd48cacfd2bde3f3aac834ef5c23b712a": "phase1-clean.tgz (docs/v2/evidence/phase1/clean/README.md:49)",
    "9e1be1705da1073e1bd5a68cc2fe5469846ead13cae5663390b4d081be5fb38e": "phase1-clean-measurements.tgz (benchmark/v0.1/GATE0_EVIDENCE_INVENTORY.md:23)",
    "3e50c16b1c950226f33cb23aaeb534a019b97da6d5c7f60880206389726cba65": "phase1-clean-curves.tgz (docs/v2/evidence/phase1/clean/README.md:50)",
    "6b823bc2249596701ec4f335c3f81c5d68db8504ae34c606f90df881dee589c5": "phase2-pilot.tgz (docs/v2/evidence/phase2/pilots/README.md:109)",
    "c882dcd1b76f071f3b47495e534acc4d6bc39fad6005ea1cbe76f709420b6bab": "phase2-pilot2.tgz (docs/v2/evidence/phase2/pilots_amendment1/README.md:65)",
    "a893840106dfd70c42cd9b9aefdaa871f73605e9fb8fec0d6c822b678b929961": "phase0a-evidence.tgz (docs/v2/evidence/phase0a/README.md:23)",
    "4fd505cd85de78701b9ca96879ce1a911fd74fdeff01e555f621eb3249c9ab85": "s2g-step4-2026-10-17.tgz, measurements only (docs/mvp-plan/evidence/step4/README.md:37)",
    "acda5d8cd4bc6560f64eba317c9cc5fd804b6e6dd195233574a8a83ac0693f2e": "phase1-pilot-v5.tgz (operator-computed, no independent expected digest)",
}


class HashingReader:
    """File wrapper that hashes every byte read, so an archive is hashed while it is streamed."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def read(self, size=-1):
        b = self.f.read(size)
        self.h.update(b)
        self.n += len(b)
        return b

    def drain(self):
        while self.read(CHUNK):
            pass
        return self.h.hexdigest()


def sha_stream(f):
    h = hashlib.sha256()
    for b in iter(lambda: f.read(CHUNK), b""):
        h.update(b)
    return h.hexdigest()


def scan(path):
    """Yield (location, digest, size) for every .pt file; return archive record if PATH is an archive."""
    path = Path(path)
    found, archive = [], None
    if path.is_dir():
        for root, _, files in os.walk(path):
            for name in sorted(files):
                if name.endswith(".pt"):
                    p = Path(root, name)
                    with p.open("rb") as f:
                        found.append((str(p), sha_stream(f), p.stat().st_size))
    elif path.is_file() and path.suffix == ".pt":
        with path.open("rb") as f:
            found.append((str(path), sha_stream(f), path.stat().st_size))
    elif path.is_file():
        with path.open("rb") as raw:
            reader = HashingReader(raw)
            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                for m in tar:
                    if m.isfile() and m.name.endswith(".pt"):
                        found.append((f"{path.name}:{m.name}", sha_stream(tar.extractfile(m)), m.size))
            digest = reader.drain()
        archive = {"path": str(path), "sha256": digest, "bytes": reader.n,
                   "recorded_as": KNOWN_BUNDLES.get(digest, "NO RECORDED DIGEST MATCH")}
    else:
        raise SystemExit(f"not found or unsupported: {path}")
    return found, archive


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--register", default=str(HERE / "SEEN_POLICIES_PROVISIONAL.csv"))
    ap.add_argument("--json", help="write an evidence record here (never the register)")
    a = ap.parse_args(argv)
    with open(a.register, newline="") as f:
        rows = list(csv.DictReader(f))
    by_digest = {r["checkpoint_sha256"]: r for r in rows if r["checkpoint_sha256"]}
    files, archives = [], []
    for p in a.paths:
        got, arch = scan(p)
        files += got
        if arch:
            archives.append(arch)
            print(f"ARCHIVE {arch['sha256']}  {arch['bytes']} bytes  {p}  -> {arch['recorded_as']}")
    matches, unregistered = {}, []
    for loc, digest, size in files:
        if digest in by_digest:
            matches.setdefault(digest, []).append(loc)
        else:
            unregistered.append((loc, digest, size))
    for digest, locs in sorted(matches.items(), key=lambda x: by_digest[x[0]]["policy_id"]):
        r = by_digest[digest]
        print(f"MATCH   {r['policy_id']:<34} {r['hash_status']:<14} {digest}  {locs[0]}" + (f" (+{len(locs) - 1} more)" if len(locs) > 1 else ""))
    for loc, digest, size in unregistered:
        print(f"UNREGISTERED {digest}  {size} bytes  {loc}")
    open_rows = [r for r in rows if not r["checkpoint_sha256"]]
    not_seen = [r for r in rows if r["checkpoint_sha256"] and r["checkpoint_sha256"] not in matches]
    print(f"\nSUMMARY: {len(files)} .pt files; {len(matches)} register rows matched by bytes; "
          f"{len(unregistered)} unregistered digests; {len(not_seen)} digest rows not seen; "
          f"{len(open_rows)} rows still without a digest (review UNREGISTERED candidates by hand)")
    if a.json:
        record = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "register_sha256": hashlib.sha256(Path(a.register).read_bytes()).hexdigest(),
                  "archives": archives,
                  "matches": {by_digest[d]["policy_id"]: {"sha256": d, "locations": l} for d, l in matches.items()},
                  "unregistered": [{"location": l, "sha256": d, "bytes": s} for l, d, s in unregistered]}
        Path(a.json).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"evidence record written to {a.json}")


if __name__ == "__main__":
    sys.exit(main())
