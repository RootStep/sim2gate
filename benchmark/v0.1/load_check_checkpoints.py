#!/usr/bin/env python3
"""CPU-only load check for recovered checkpoints (no simulator, no GPU, no evaluation).

Usage (inside the Arena container, where torch is installed):
    /isaac-sim/python.sh load_check_checkpoints.py FILE.pt [FILE.pt ...] --register CSV --expect ID [ID ...] --json OUT

For each file: SHA-256 of the bytes, the register row it matches, torch.load on CPU (weights_only first),
top-level keys, training iteration if recorded, tensor and parameter counts, whether every tensor is finite, and a
structure fingerprint of the non-optimizer tensors (names, shapes, dtypes). PASS needs: digest registered under an
expected ID, load succeeded, all tensors finite, and the same structure fingerprint as every other file.
This proves the bytes deserialize into a consistent network; it does not run a policy.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensors(obj, prefix=""):
    import torch
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from tensors(obj[k], f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from tensors(v, f"{prefix}/{i}")


def load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
    except Exception as e:  # older checkpoints may hold non-tensor Python objects
        note = f"weights_only failed: {type(e).__name__}"
    return torch.load(path, map_location="cpu", weights_only=False), note


def check_file(path, by_digest):
    rec = {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": sha256(path)}
    row = by_digest.get(rec["sha256"])
    rec["policy_id"] = row["policy_id"] if row else None
    rec["register_status"] = row["hash_status"] if row else None
    try:
        obj, mode = load(path)
    except Exception as e:
        rec.update(loaded=False, error=f"{type(e).__name__}: {e}")
        return rec
    rec["loaded"], rec["load_mode"] = True, mode
    rec["top_level_keys"] = sorted(map(str, obj)) if isinstance(obj, dict) else type(obj).__name__
    rec["iter"] = obj.get("iter") if isinstance(obj, dict) else None
    ts = list(tensors(obj))
    rec["tensors"] = len(ts)
    rec["finite"] = all(bool(t.isfinite().all()) for _, t in ts if t.is_floating_point())
    model = [(k, tuple(t.shape), str(t.dtype)) for k, t in ts if "optimizer" not in k]
    rec["parameters"] = sum(math.prod(s) for _, s, _ in model)
    rec["structure_sha256"] = hashlib.sha256(json.dumps(model).encode()).hexdigest()
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--register", required=True)
    ap.add_argument("--expect", nargs="+", required=True, help="register policy_ids that must all be present")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    with open(a.register, newline="") as f:
        by_digest = {r["checkpoint_sha256"]: r for r in csv.DictReader(f) if r["checkpoint_sha256"]}
    recs = [check_file(p, by_digest) for p in a.files]
    structures = {r.get("structure_sha256") for r in recs if r.get("loaded")}
    seen = {r["policy_id"] for r in recs if r["policy_id"]}
    expect = set(a.expect)
    ok = True
    for r in sorted(recs, key=lambda r: str(r["policy_id"])):
        problems = []
        if r["policy_id"] not in expect:
            problems.append("digest not registered under an expected ID")
        if not r.get("loaded"):
            problems.append(r.get("error", "load failed"))
        elif not r["finite"]:
            problems.append("non-finite tensor")
        if r.get("loaded") and len(structures) > 1:
            problems.append("structure differs between files")
        r["result"] = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        ok &= not problems
        print(f"{r['result'][:4]}  {str(r['policy_id']):<12} iter={r.get('iter')} tensors={r.get('tensors')} "
              f"params={r.get('parameters')} {r.get('load_mode', '')}  {r['path']}" +
              ("" if r["result"] == "PASS" else f"  <- {r['result']}"))
    missing = sorted(expect - seen)
    if missing:
        ok = False
        print("MISSING expected IDs:", ", ".join(missing))
    print(f"\nSUMMARY: {sum(r['result'] == 'PASS' for r in recs)}/{len(expect)} expected checkpoints PASS; "
          f"{len(structures)} distinct network structure(s); {'ALL PASS' if ok else 'NOT PASSED'}")
    if a.json:
        Path(a.json).write_text(json.dumps({"tool_sha256": sha256(__file__), "register_sha256": sha256(a.register),
                                            "expected": sorted(expect), "missing": missing, "all_pass": ok,
                                            "files": recs}, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
