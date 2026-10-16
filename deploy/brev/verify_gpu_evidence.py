#!/usr/bin/env python3
"""Verify a GPU evidence folder against what the review requires, and write the verdict into the folder.

    python3 verify_gpu_evidence.py <folder> --commit <full sha> --checkpoint-sha256 <sha> [--outside <file>]

Reads, from <folder>: sim2gate-versions.txt, sim2gate_smoke_identity/report.json, sim2gate_check2_eta1_v2.json,
and, when given, the outside probe's saved output. Writes VERIFY.json and prints one line per item:

- pinned commit:   the host checkout and the installed package are both exactly --commit
- eta 0 identity, final checkpoints: status from the smoke test; passed only if two stock runs match each other
                   AND the eta 0 run matches stock, tensor by tensor (model and optimizer); inconclusive when stock
                   training is not reproducible on the device
- eta 0 identity, one update: sim2gate_update_identity.json (python -m sim2gate.training.isaaclab_launch
                   --update-identity); the same PPO update replayed from one saved state, stock twice and with
                   Sim2Gate attached; inconclusive if missing or if the stock replays differ
- smoke test:      the other smoke checks passed
- contact force:   check 2 measured the intended policy, every physics substep (samples == decimation, covered
                   share 1.0) with the per-update sampling method, for the full 1000 steps
- ingress:         the outside probe ran, probed at least one port, and found none open (exit code 0). The one
                   exception: port 22 may be open when --sshd shows SSH accepts keys only (password and
                   keyboard-interactive authentication off, public-key on); it is then listed as an allowed exception

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


def check_update_identity(folder):
    doc, err = _load(os.path.join(folder, "sim2gate_update_identity.json"))
    if err:
        return "inconclusive", f"not run ({err})"
    status = doc.get("status")
    complete = all(k in doc for k in ("stock_vs_stock", "stock_vs_sim2gate", "monitor_records_written"))
    if not complete or status not in ("passed", "failed", "inconclusive"):
        return "failed", {"problem": "incomplete update identity report", "status": status}
    return status, {k: doc.get(k) for k in ("stock_vs_stock", "stock_vs_sim2gate", "monitor_records_written",
                                            "num_transitions", "device", "cublas_workspace_config", "not_covered",
                                            "note")}


def check_contact(folder, ck_sha, steps, name="sim2gate_check2_eta1_v2.json"):
    rep, err = _load(os.path.join(folder, name))
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
              "play_args": rep.get("play_args"),
              "metrics": rep.get("metrics"), "problems": problems}
    return ("failed" if problems else "passed"), detail


SSHD_REQUIRED = {"passwordauthentication": "no", "kbdinteractiveauthentication": "no", "pubkeyauthentication": "yes"}


def sshd_key_only(path):
    """(True, settings) if `sshd -T` output shows key-only authentication; (False, reason) otherwise."""
    if not path:
        return False, "no sshd settings supplied"
    try:
        text = open(path).read()
    except OSError as e:
        return False, str(e)
    got = {}
    for line in text.splitlines():
        parts = line.strip().lower().split()
        if len(parts) == 2:
            got[parts[0]] = parts[1]
    bad = {k: got.get(k) for k, v in SSHD_REQUIRED.items() if got.get(k) != v}
    if bad:
        return False, {"required": SSHD_REQUIRED, "found": bad}
    return True, {k: got[k] for k in sorted(got)}


def check_outside(path, sshd=None):
    if not path:
        return "inconclusive", "outside probe output not supplied"
    try:
        text = open(path).read()
    except OSError as e:
        return "failed", str(e)
    total = re.search(r"^# (\d+) of (\d+) probed TCP ports reachable$", text, re.M)
    code = re.search(r"^exit code (\d+)$", text, re.M)
    opened = re.findall(r"^tcp (\d+) OPEN$", text, re.M)
    intercepted = re.findall(r"^tcp (\d+) INTERCEPTED-LOCALLY$", text, re.M)
    probed = re.findall(r"^tcp (\d+) (?:OPEN|closed-or-filtered|INTERCEPTED-LOCALLY)$", text, re.M)
    controlled = "# control:" in text
    detail = {"probed_ports": probed, "open_ports": opened, "intercepted_ports": intercepted,
              "control_probe": controlled, "summary": total and total.group(0),
              "exit_code": code and int(code.group(1))}
    if not total or not code:
        return "failed", {**detail, "problem": "probe summary or saved exit code missing"}
    n_open, n_probed, rc = int(total.group(1)), int(total.group(2)), int(code.group(1))
    want_rc = 1 if n_open else 3 if intercepted else 0
    if n_probed == 0 or n_probed != len(probed) or n_open != len(opened) or rc != (want_rc if controlled else
                                                                                   (1 if n_open else 0)):
        return "failed", {**detail, "problem": "probe output is inconsistent or probed nothing"}
    unexpected = [q for q in opened if q != "22"]
    if "22" in opened:
        ok, why = sshd_key_only(sshd)
        detail["ssh_port_22"] = {"allowed": ok, "sshd": why}
        if ok:
            detail["allowed_exceptions"] = ["22 (SSH, key-only authentication; used by brev shell and scp)"]
        else:
            unexpected.append("22")
    if unexpected:
        detail["unexpected_open_ports"] = unexpected
        if not controlled:
            detail["note"] = ("probe ran without the interception control (older check_exposure.sh); an OPEN port "
                              "may be answered by the probing network itself")
        return "failed", detail
    if intercepted:
        detail["note"] = (f"ports {', '.join(intercepted)} are answered by the probing network itself, so their "
                          "exposure could not be measured from there")
        return "inconclusive", detail
    return "passed", detail


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder")
    p.add_argument("--commit", required=True, help="full 40-character sim2gate commit the run must use")
    p.add_argument("--checkpoint-sha256", required=True, help="SHA-256 of the policy check 2 must measure")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--check2", default="sim2gate_check2_eta1_v2.json", help="check 2 report file in the folder")
    p.add_argument("--only", default=None,
                   help="semicolon-separated item names to check, for evidence split across runs (the pinned commit "
                        "is always checked); VERIFY.json lists the items left out")
    p.add_argument("--outside", default=None, help="saved output of check_exposure.sh --probe, with its exit code")
    p.add_argument("--sshd", default=None, help="saved `sshd -T` authentication settings, needed if port 22 is open")
    a = p.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        p.error("--commit must be the full 40-character SHA")
    ident, smoke = check_smoke(a.folder)
    items = {"pinned commit": check_commit(a.folder, a.commit), "eta 0 identity, final checkpoints": ident,
             "eta 0 identity, one update": check_update_identity(a.folder), "smoke test": smoke,
             "contact force 4/4": check_contact(a.folder, a.checkpoint_sha256, a.steps, a.check2),
             "ingress from outside": check_outside(a.outside, a.sshd)}
    skipped = []
    if a.only:
        want = {x.strip() for x in a.only.split(";") if x.strip()} | {"pinned commit"}
        unknown = want - set(items)
        if unknown:
            p.error(f"--only: unknown items {sorted(unknown)}; choose from {sorted(items)}")
        skipped = [k for k in items if k not in want]
        items = {k: v for k, v in items.items() if k in want}
    statuses = [s for s, _ in items.values()]
    overall = "failed" if "failed" in statuses else "inconclusive" if "inconclusive" in statuses else "passed"
    out = {"overall": overall, "items": {k: {"status": s, "detail": d} for k, (s, d) in items.items()},
           "not_checked_here": skipped,
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
