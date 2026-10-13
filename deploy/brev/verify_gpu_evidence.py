#!/usr/bin/env python3
"""Verify a GPU evidence folder against what the review requires, and write the verdict into the folder.

    python3 verify_gpu_evidence.py <folder> --commit <full sha> --checkpoint-sha256 <sha> [--outside <file>]

Reads, from <folder>: sim2gate-versions.txt, sim2gate_smoke_identity/report.json, sim2gate_check2_eta1_v2.json,
and, when given, the outside probe's saved output. Writes VERIFY.json and prints one line per item:

- pinned commit:   the host checkout and the installed package are both exactly --commit
- eta 0 identity:  status from the smoke test; passed only if two stock runs match each other AND the eta 0 run
                   matches stock, tensor by tensor (model and optimizer); inconclusive when stock is not reproducible
- smoke test:      the other smoke checks passed
- contact force:   check 2 measured the intended policy, every physics substep (samples == decimation, covered
                   share 1.0) with the per-update sampling method, for the full 1000 steps
- ingress:         the outside probe ran, probed at least one port, found none open, and exited 0

Exit code: 0 all passed, 1 any failed, 3 none failed but something is inconclusive. Standard library only.
"""
import argparse
import json
import os
import re
import sys


def _load(path):
    try:
        with open(path) as f:
            return json.load(f), None
    except (OSError, ValueError) as e:
        return None, f"{os.path.basename(path)}: {e}"


def check_commit(folder, commit):
    p = os.path.join(folder, "sim2gate-versions.txt")
    try:
        text = open(p).read()
    except OSError as e:
        return "failed", str(e)
    host = re.search(r"^sim2gate_commit=(\S+)$", text, re.M)
    pkg = re.search(r"^sim2gate \S+ (\S+)$", text, re.M)
    got = {"host_checkout": host and host.group(1), "installed_package": pkg and pkg.group(1)}
    ok = got["host_checkout"] == commit and got["installed_package"] == commit
    return ("passed" if ok else "failed"), {"expected": commit, **got}


def check_smoke(folder):
    rep, err = _load(os.path.join(folder, "sim2gate_smoke_identity", "report.json"))
    if err:
        return ("failed", err), ("failed", err)
    checks = rep.get("checks") or []
    ident = [c for c in checks if c.get("name", "").startswith("eta 0 with Sim2Gate")]
    if len(ident) != 1:
        idv = ("failed", "identity check missing from the smoke report (was --identity passed?)")
    else:
        c = ident[0]
        d = c.get("detail") or {}
        complete = isinstance(d, dict) and all(k in d for k in ("stock_vs_repeat", "stock_vs_eta0", "checkpoints"))
        status = c.get("status") if complete else "failed"
        idv = (status if status in ("passed", "failed", "inconclusive") else "failed",
               {k: d.get(k) for k in ("stock_vs_repeat", "stock_vs_eta0", "note")} if complete else d)
    others = [c for c in checks if c not in ident]
    bad = [c["name"] for c in others if not c.get("ok")]
    smoke = ("failed" if bad or not others else "passed",
             {"failed_checks": bad, "checks": len(others), "result": rep.get("result")})
    return idv, smoke


def check_contact(folder, ck_sha, steps):
    rep, err = _load(os.path.join(folder, "sim2gate_check2_eta1_v2.json"))
    if err:
        return "failed", err
    cs = rep.get("contact_force_substeps") or {}
    ck = rep.get("checkpoint") or {}
    n, dec = cs.get("samples_per_control_step"), cs.get("physics_substeps_per_control_step")
    problems = []
    if ck.get("sha256") != ck_sha:
        problems.append(f"measured checkpoint sha256 {ck.get('sha256')}, expected {ck_sha}")
    if not dec or n != dec or cs.get("covered_share") != 1.0:
        problems.append(f"contact force covered {n} of {dec} physics substeps (share {cs.get('covered_share')})")
    if "every scene update" not in str(cs.get("method", "")):
        problems.append("report does not record the per-update sampling method (older check 2?)")
    if rep.get("control_steps") != steps:
        problems.append(f"control_steps {rep.get('control_steps')}, expected {steps}")
    detail = {"samples_per_control_step": n, "physics_substeps_per_control_step": dec,
              "covered_share": cs.get("covered_share"), "physics_handles_decimation": cs.get("physics_handles_decimation"),
              "scene_updates_per_step_histogram": cs.get("scene_updates_per_step_histogram"),
              "method": cs.get("method"), "checkpoint": ck, "control_steps": rep.get("control_steps"),
              "metrics": rep.get("metrics"), "problems": problems}
    return ("failed" if problems else "passed"), detail


def check_outside(path):
    if not path:
        return "inconclusive", "outside probe output not supplied"
    try:
        text = open(path).read()
    except OSError as e:
        return "failed", str(e)
    total = re.search(r"^# (\d+) of (\d+) probed TCP ports reachable$", text, re.M)
    code = re.search(r"^exit code (\d+)$", text, re.M)
    opened = re.findall(r"^tcp (\d+) OPEN$", text, re.M)
    detail = {"open_ports": opened, "summary": total and total.group(0), "exit_code": code and int(code.group(1))}
    if not total or not code:
        return "failed", {**detail, "problem": "probe summary or saved exit code missing"}
    ok = int(total.group(1)) == 0 and int(total.group(2)) > 0 and not opened and int(code.group(1)) == 0
    return ("passed" if ok else "failed"), detail


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder")
    p.add_argument("--commit", required=True, help="full 40-character sim2gate commit the run must use")
    p.add_argument("--checkpoint-sha256", required=True, help="SHA-256 of the policy check 2 must measure")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--outside", default=None, help="saved output of check_exposure.sh --probe, with its exit code")
    a = p.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        p.error("--commit must be the full 40-character SHA")
    ident, smoke = check_smoke(a.folder)
    items = {"pinned commit": check_commit(a.folder, a.commit), "eta 0 identity": ident, "smoke test": smoke,
             "contact force 4/4": check_contact(a.folder, a.checkpoint_sha256, a.steps),
             "ingress from outside": check_outside(a.outside)}
    statuses = [s for s, _ in items.values()]
    overall = "failed" if "failed" in statuses else "inconclusive" if "inconclusive" in statuses else "passed"
    out = {"overall": overall, "items": {k: {"status": s, "detail": d} for k, (s, d) in items.items()},
           "inputs": {"commit": a.commit, "checkpoint_sha256": a.checkpoint_sha256, "steps": a.steps,
                      "outside": a.outside and os.path.basename(a.outside)}}
    with open(os.path.join(a.folder, "VERIFY.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, (s, d) in items.items():
        print(f"{s.upper():13s} {k}")
    print(f"{overall.upper():13s} overall  (details: {os.path.join(a.folder, 'VERIFY.json')})")
    return {"passed": 0, "failed": 1, "inconclusive": 3}[overall]


if __name__ == "__main__":
    sys.exit(main())
