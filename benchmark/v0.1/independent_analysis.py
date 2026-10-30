#!/usr/bin/env python3
"""Tier-R independent truth analysis (offline, CPU, standard library). Reads the collector's run files and labels
each policy with independent_truth.py only. No Sim2Gate code or limits are imported.

    python benchmark/v0.1/independent_analysis.py --manifest G2.json --runs RUNS_DIR --out-json OUT.json --out-md OUT.md

RUNS_DIR holds `<policy>__<cell>.jsonl.gz` from independent_collector.py. For each policy x cell: episode count,
falls, tracking losses, failures (primary, consequence-based), and secondary slip magnitude (median over
episodes of the stance-path slip ratio and of the per-episode p90 stance path). For each non-control policy and
cell, manifestation vs the pooled controls in the same cell (one-sided Fisher exact, minimum rate difference).
A policy's primary truth = manifested in at least one of its arm's primary cells, at alpha divided by the number of
primary cells (Bonferroni).

Run validity (revision 2): a run is INVALID, and excluded from every count including the pooled controls, if its
header is not marked completed, its footer is missing or does not match the records, it used another manifest,
checkpoint, cell, command set or task, its runtime verification or step-synchronization evidence is missing or failed, any episode ended before the
time limit or is not exactly episode_steps long, or any episode fails the schema. Invalid and missing runs are
reported separately and never imputed. If any required run (each non-control policy in its arm's primary cells,
each control in every primary cell) is missing or invalid, the status is INCOMPLETE, no primary truth label is
issued for any policy, and the program exits 2.
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


def read_run(path, manifest=None, manifest_sha=None, policy_id=None, cell_id=None):
    """Read one run file and check it. Returns (header, episodes, problems); any problem makes the run INVALID,
    and an invalid run is excluded from every count (its own cell and the pooled controls)."""
    problems = []
    try:
        with gzip.open(path, "rt") as f:
            lines = [json.loads(line) for line in f if line.strip()]
    except (OSError, EOFError, ValueError) as e:
        return None, [], [f"unreadable: {type(e).__name__}"]
    if not lines or "header" not in lines[0]:
        return None, [], ["no header"]
    header = lines[0]["header"]
    footer = lines[-1].get("footer") if len(lines) > 1 else None
    episodes = [x for x in lines[1:] if "footer" not in x]
    if header.get("completed") is not True:
        problems.append("header not marked completed")
    if footer is None:
        problems.append("no footer (truncated or incomplete run)")
    else:
        h = hashlib.sha256()
        for ep in episodes:
            h.update(json.dumps(ep, separators=(",", ":")).encode())
        if footer.get("records") != len(episodes) or footer.get("records_sha256") != h.hexdigest():
            problems.append("footer does not match the records")
    if manifest_sha and header.get("manifest_sha256") != manifest_sha:
        problems.append("run used a different manifest")
    if policy_id is not None and header.get("policy_id") != policy_id:
        problems.append("header policy differs from the file name")
    if cell_id is not None and (header.get("cell") or {}).get("id") != cell_id:
        problems.append("header cell differs from the file name")
    if header.get("checkpoint_matches_manifest") is not True:
        problems.append("checkpoint not confirmed against the manifest")
    if manifest is not None:
        pol = manifest["policies"].get(policy_id) if policy_id else None
        if pol and (header.get("checkpoint") or {}).get("sha256") != pol["checkpoint_sha256"]:
            problems.append("checkpoint digest differs from the manifest")
        cells = {c["id"]: c for c in manifest["cells"]}
        if cell_id in cells and header.get("cell") != cells[cell_id]:
            problems.append("cell definition differs from the manifest")
        if header.get("command_set_sha256") != manifest["command_set"]["sha256"]:
            problems.append("command set differs from the manifest")
        if header.get("task") != manifest["common_task"]:
            problems.append("task is not the common task")
        n_expected = int(manifest["episodes_per_cell_repeat"]) * int(manifest["command_set"].get("n_commands", 64))
        if header.get("num_envs") != n_expected:
            problems.append(f"num_envs {header.get('num_envs')} != {n_expected}")
    if (header.get("runtime_checks") or {}).get("ok") is not True:
        problems.append("runtime verification missing or failed")
    if (header.get("sync_checks") or {}).get("ok") is not True:
        problems.append("step synchronization evidence missing or failed")
    if header.get("unexpected_early_dones", 1) != 0:
        problems.append("episodes ended before the time limit")
    if len(episodes) != header.get("num_envs"):
        problems.append("episode count differs from num_envs")
    steps = header.get("episode_steps")
    bad_eps, short = 0, 0
    for ep in episodes:
        try:
            n = it.validate_episode(ep)
        except (it.SchemaError, ValueError, TypeError, KeyError):
            bad_eps += 1
            continue
        if steps is None or n != steps:
            short += 1
    if bad_eps:
        problems.append(f"{bad_eps} episodes fail the schema")
    if short:
        problems.append(f"{short} episodes are not exactly episode_steps long")
    return header, episodes, problems


def required_runs(manifest):
    """(policy, cell) pairs without which the primary result is not defined: every non-control policy in its arm's
    primary cells, and every control in every primary cell (they form the pooled reference)."""
    pol = manifest["policies"]
    prim = manifest["primary_cells"]
    all_primary = sorted({c for cs in prim.values() for c in cs})
    req = set()
    for pid, meta in pol.items():
        cells = all_primary if meta["arm"] == "control" else prim.get(meta["arm"], [])
        req.update((pid, c) for c in cells)
    return req


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
    table, status, run_problems = {}, {}, {}
    for pid in policies:
        for cid in cells:
            path = Path(runs_dir) / f"{pid}__{cid}.jsonl.gz"
            table[(pid, cid)] = None
            if not path.exists():
                status[(pid, cid)] = "missing"
                continue
            header, eps, probs = read_run(path, manifest, manifest_sha, pid, cid)
            if probs:
                status[(pid, cid)] = "invalid"
                run_problems[f"{pid}__{cid}"] = probs
                continue
            status[(pid, cid)] = "ok"
            table[(pid, cid)] = summarize_cell(eps, rules)
    pooled = {}
    for cid in cells:
        bad = [p for p in controls if status[(p, cid)] != "ok"]
        pooled[cid] = None if bad else {
            "failures": sum(table[(p, cid)]["failures"] for p in controls),
            "valid": sum(table[(p, cid)]["valid"] for p in controls)}
    req = required_runs(manifest)
    req_missing = sorted(f"{p}__{c}" for p, c in req if status[(p, c)] == "missing")
    req_invalid = sorted(f"{p}__{c}" for p, c in req if status[(p, c)] == "invalid")
    complete = not req_missing and not req_invalid
    counts = {k: sum(1 for v in status.values() if v == k) for k in ("ok", "invalid", "missing")}
    result = {"status": "COMPLETE" if complete else "INCOMPLETE", "rules": rules, "run_counts": counts,
              "required_missing": req_missing, "required_invalid": req_invalid,
              "pooled_controls": pooled, "run_problems": run_problems, "policies": {}}
    for pid, meta in policies.items():
        arm = meta["arm"]
        primary = manifest["primary_cells"].get(arm, [])
        alpha = rules["alpha"] / max(1, len(primary))
        rules_cell = dict(rules, alpha=alpha)
        per_cell = {}
        for cid in cells:
            s = table[(pid, cid)]
            entry = {"status": status[(pid, cid)], "summary": s}
            if s is not None and arm != "control" and pooled[cid] is not None and s["valid"] and pooled[cid]["valid"]:
                entry["vs_controls"] = it.manifestation(s["failures"], s["valid"], pooled[cid]["failures"],
                                                        pooled[cid]["valid"], rules_cell)
            per_cell[cid] = entry
        unavailable = [c for c in primary if "vs_controls" not in per_cell[c]]
        truth = None
        if arm != "control" and complete and not unavailable:
            truth = any(per_cell[c]["vs_controls"]["manifested"] for c in primary)
        cross = sorted(c for c in cells if c not in primary and per_cell[c].get("vs_controls", {}).get("manifested"))
        result["policies"][pid] = {"arm": arm, "primary_cells": primary, "alpha_per_primary_cell": alpha,
                                   "primary_truth_manifested": truth, "unavailable_primary_cells": unavailable,
                                   "cross_symptom_cells": cross, "cells": per_cell}
    return result


def to_markdown(result, cells):
    c = result["run_counts"]
    lines = [f"**Status: {result['status']}** — runs ok {c['ok']}, invalid {c['invalid']}, missing {c['missing']}.", ""]
    if result["status"] != "COMPLETE":
        lines += ["Required primary runs are missing or invalid, so no policy has a primary truth label "
                  f"(missing: {', '.join(result['required_missing']) or 'none'}; "
                  f"invalid: {', '.join(result['required_invalid']) or 'none'}).", ""]
    lines += ["| policy | arm | primary truth | " + " | ".join(f"{c} fail/n" for c in cells) + " | cross symptoms |",
             "|---|---|---|" + "---|" * len(cells) + "---|"]
    for pid, r in result["policies"].items():
        t = r["primary_truth_manifested"]
        ts = "control" if r["arm"] == "control" else ("missing" if t is None else ("MANIFESTED" if t else "not manifested"))
        cols = []
        for c in cells:
            s = r["cells"][c]["summary"]
            if s is None:
                cols.append(r["cells"][c]["status"])
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
    if result["status"] != "COMPLETE":
        print("[independent] INCOMPLETE: required primary runs missing or invalid; no truth labels issued",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
