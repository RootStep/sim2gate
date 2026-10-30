#!/usr/bin/env python3
"""Tier-R independent truth analysis (offline, CPU, standard library). Reads the collector's run files and labels
each policy with independent_truth.py only. No Sim2Gate code or limits are imported.

    python benchmark/v0.1/independent_analysis.py --manifest G2.json --runs RUNS_DIR --out-json OUT.json --out-md OUT.md

RUNS_DIR holds `<policy>__<cell>.jsonl.gz` from independent_collector.py. For each policy x cell: episode count,
falls, tracking losses, failures (primary, consequence-based), and secondary slip magnitude (median over
episodes of the stance-path slip ratio and of the per-episode p90 stance path). For each non-control policy and
cell, manifestation vs the pooled controls in the same cell (one-sided Fisher exact, minimum rate difference).
A policy's primary truth = manifested in at least one of its arm's primary cells, at alpha divided by the number of
primary cells (Bonferroni). Missing runs are reported as missing, never imputed.
"""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import independent_truth as it  # noqa: E402


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_run(path, manifest_sha=None):
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())["header"]
        episodes = [json.loads(line) for line in f if line.strip()]
    problems = []
    if manifest_sha and header.get("manifest_sha256") != manifest_sha:
        problems.append("run used a different manifest")
    if header.get("checkpoint_matches_manifest") is not True:
        problems.append("checkpoint not confirmed against the manifest")
    if len(episodes) != header.get("num_envs"):
        problems.append("episode count differs from num_envs")
    return header, episodes, problems


def summarize_cell(episodes, rules):
    out = {"episodes": len(episodes), "falls": 0, "tracking_losses": 0, "failures": 0, "invalid": 0,
           "slip_ratio_median": None, "stance_p90_median_m": None}
    ratios, p90s = [], []
    for ep in episodes:
        try:
            o = it.episode_outcome(ep, rules)
        except (it.SchemaError, ValueError):
            out["invalid"] += 1
            continue
        out["falls"] += o["fell"]
        out["tracking_losses"] += o["tracking_loss"]
        out["failures"] += o["failure"]
        s = it.stance_slip(ep, rules, end=o["fall_step"])
        if s["slip_ratio"] is not None:
            ratios.append(s["slip_ratio"])
        if s["p90_m"] is not None:
            p90s.append(s["p90_m"])
    out["valid"] = out["episodes"] - out["invalid"]
    out["slip_ratio_median"] = it._quantile(ratios, 0.5)
    out["stance_p90_median_m"] = it._quantile(p90s, 0.5)
    return out


def analyse(manifest, manifest_sha, runs_dir):
    rules = dict(manifest["truth_rules"])
    policies, cells = manifest["policies"], [c["id"] for c in manifest["cells"]]
    controls = [p for p, v in policies.items() if v["arm"] == "control"]
    table, run_problems = {}, {}
    for pid in policies:
        for cid in cells:
            path = Path(runs_dir) / f"{pid}__{cid}.jsonl.gz"
            if not path.exists():
                table[(pid, cid)] = None
                continue
            header, eps, probs = read_run(path, manifest_sha)
            if probs:
                run_problems[f"{pid}__{cid}"] = probs
            table[(pid, cid)] = summarize_cell(eps, rules)
    pooled = {}
    for cid in cells:
        cs = [table[(p, cid)] for p in controls if table[(p, cid)] is not None]
        pooled[cid] = None if len(cs) != len(controls) else {
            "failures": sum(c["failures"] for c in cs), "valid": sum(c["valid"] for c in cs)}
    result = {"rules": rules, "pooled_controls": pooled, "run_problems": run_problems, "policies": {}}
    for pid, meta in policies.items():
        arm = meta["arm"]
        primary = manifest["primary_cells"].get(arm, [])
        alpha = rules["alpha"] / max(1, len(primary))
        rules_cell = dict(rules, alpha=alpha)
        per_cell = {}
        for cid in cells:
            s = table[(pid, cid)]
            entry = {"summary": s}
            if s is not None and arm != "control" and pooled[cid] is not None and s["valid"] and pooled[cid]["valid"]:
                entry["vs_controls"] = it.manifestation(s["failures"], s["valid"], pooled[cid]["failures"],
                                                        pooled[cid]["valid"], rules_cell)
            per_cell[cid] = entry
        missing_primary = [c for c in primary if table[(pid, c)] is None or "vs_controls" not in per_cell[c]]
        truth = None
        if arm != "control" and not missing_primary:
            truth = any(per_cell[c]["vs_controls"]["manifested"] for c in primary)
        cross = sorted(c for c in cells if c not in primary and per_cell[c].get("vs_controls", {}).get("manifested"))
        result["policies"][pid] = {"arm": arm, "primary_cells": primary, "alpha_per_primary_cell": alpha,
                                   "primary_truth_manifested": truth, "missing_primary_cells": missing_primary,
                                   "cross_symptom_cells": cross, "cells": per_cell}
    return result


def to_markdown(result, cells):
    lines = ["| policy | arm | primary truth | " + " | ".join(f"{c} fail/n" for c in cells) + " | cross symptoms |",
             "|---|---|---|" + "---|" * len(cells) + "---|"]
    for pid, r in result["policies"].items():
        t = r["primary_truth_manifested"]
        ts = "control" if r["arm"] == "control" else ("missing" if t is None else ("MANIFESTED" if t else "not manifested"))
        cols = []
        for c in cells:
            s = r["cells"][c]["summary"]
            if s is None:
                cols.append("missing")
                continue
            v = r["cells"][c].get("vs_controls")
            mark = "*" if v and v["manifested"] else ""
            cols.append(f"{s['failures']}/{s['valid']}{mark}")
        lines.append(f"| {pid} | {r['arm']} | {ts} | " + " | ".join(cols) + f" | {', '.join(r['cross_symptom_cells']) or '—'} |")
    return "\n".join(lines) + "\n\n`*` = manifested vs pooled controls in that cell (Bonferroni-adjusted alpha for primary cells).\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md")
    a = ap.parse_args(argv)
    with open(a.manifest) as fh:
        manifest = json.load(fh)
    result = analyse(manifest, _sha(a.manifest), a.runs)
    result.update(tool_sha256=_sha(__file__), truth_lib_sha256=_sha(HERE / "independent_truth.py"),
                  manifest_sha256=_sha(a.manifest), tier="R (corrected retrospective, post hoc, development only)")
    Path(a.out_json).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    md = to_markdown(result, [c["id"] for c in manifest["cells"]])
    if a.out_md:
        Path(a.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
