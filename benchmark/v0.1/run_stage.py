#!/usr/bin/env python3
"""Tier-R stage driver, run on the Brev HOST (standard library only). One collector run per (policy, cell) of an
APPROVED stage, each in the container via `docker exec`, serially.

    python3 run_stage.py --stage S1 --preflight     # no GPU: SHA-256 of every manifest checkpoint, on the instance
    python3 run_stage.py --stage S1 --run           # the stage's runs (needs the stage APPROVED in the manifest)

--run, per run in manifest order:
  - an existing output that passes read_run is kept and skipped; one that does not is moved to invalid/;
  - before starting a run, stop if the cumulative collector wall time plus the per-run estimate would pass the
    stage's gpu_hour_cap (accounting decided by the manager on 2026-10-30: summed collector wall time);
  - the collector runs under a per-run timeout; exit 0 = done; any other exit is retried once from scratch (a
    file left by a step-synchronization failure, exit 3, is moved to invalid/ first); a second failure is recorded
    and the stage continues with the next run;
  - every run is logged to <output_dir>/stage_<id>.jsonl (start, end, wall seconds, exit, attempt).
At the end it re-validates every output with independent_analysis.read_run and writes stage_<id>_summary.json.
It computes NO outcomes (falls, failures, manifestation): outcome analysis happens only after S2 completes.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import independent_analysis as ia  # noqa: E402

PER_RUN_TIMEOUT_S = 600
PER_RUN_ESTIMATE_S = 69 * 1.5            # S0 attempt 4 measured 69 s; 1.5x margin for the stop-before-cap check


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Driver:
    def __init__(self, a):
        self.a = a
        self.manifest_path = os.path.join(HERE, "G2_TIER_R_MANIFEST.json")
        self.m = json.load(open(self.manifest_path))
        self.msha = ia._sha(self.manifest_path)
        stages = {s["id"]: s for s in self.m["stages"]}
        if a.stage not in stages:
            raise SystemExit(f"stage {a.stage!r} not in the manifest")
        self.st = stages[a.stage]
        self.out_dir_rel = self.st["output_dir"]
        self.out_dir = os.path.join(a.host_root, self.out_dir_rel)
        self.tools_rel = os.path.relpath(HERE, a.host_root) if HERE.startswith(a.host_root) else a.tools_rel

    def docker(self, args, **kw):
        return subprocess.run([self.a.docker, "exec", "-w", self.a.container_root, self.a.container] + args,
                              stdin=subprocess.DEVNULL, **kw)

    # ---- preflight: checkpoint bytes on the instance -------------------------------------------------------------
    def preflight(self):
        pol = self.m["policies"]
        needed = sorted({p for p, _ in self.st["runs"]}) if self.a.stage_only else sorted(pol)
        rows, bad = [], 0
        for pid in needed:
            path = os.path.join(self.a.container_root, pol[pid]["checkpoint"])
            r = self.docker(["sha256sum", path], capture_output=True, text=True)
            got = r.stdout.split()[0] if r.returncode == 0 and r.stdout else None
            ok = got == pol[pid]["checkpoint_sha256"]
            bad += not ok
            rows.append({"policy": pid, "path": path, "expected": pol[pid]["checkpoint_sha256"], "got": got,
                         "status": "MATCH" if ok else ("MISSING" if got is None else "MISMATCH")})
            print(f"{rows[-1]['status']:8s} {pid:12s} {got or r.stderr.strip()[:80]}")
        rec = {"utc": now(), "stage": self.a.stage, "manifest_sha256": self.msha, "checked": len(rows),
               "mismatches": bad, "rows": rows}
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, f"preflight_{self.a.stage}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"== preflight: {len(rows) - bad}/{len(rows)} checkpoints MATCH the manifest")
        return 0 if bad == 0 else 1

    # ---- run -----------------------------------------------------------------------------------------------------
    def log(self, rec):
        with open(os.path.join(self.out_dir, f"stage_{self.a.stage}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def used_seconds(self):
        p = os.path.join(self.out_dir, f"stage_{self.a.stage}.jsonl")
        if not os.path.exists(p):
            return 0.0
        return sum(json.loads(x).get("wall_seconds", 0.0) for x in open(p) if x.strip())

    def quarantine(self, path, why):
        d = os.path.join(self.out_dir, "invalid")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, os.path.basename(path) + "." + datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
        shutil.move(path, dst)
        self.log({"utc": now(), "event": "quarantined", "file": dst, "reason": why})

    def valid(self, path, pid, cid):
        _, _, probs = ia.read_run(path, self.m, self.msha, pid, cid)
        return probs

    def collect(self, pid, cid, attempt):
        out_rel = f"{self.out_dir_rel}/{pid}__{cid}.jsonl.gz"
        ckpt = os.path.join(self.a.container_root, self.m["policies"][pid]["checkpoint"])
        logs = os.path.join(self.out_dir, "logs")
        os.makedirs(logs, exist_ok=True)
        py = "/isaac-sim/python.sh"
        args = ["timeout", str(PER_RUN_TIMEOUT_S), py, f"{self.tools_rel}/independent_collector.py",
                "--manifest", f"{self.tools_rel}/G2_TIER_R_MANIFEST.json", "--stage", self.a.stage,
                "--policy", pid, "--cell", cid, "--out", out_rel, "--",
                "--task", self.m["common_task"], "--checkpoint", ckpt] + list(self.m["play_args_fixed"])
        start, t0 = now(), time.time()
        with open(os.path.join(logs, f"{pid}__{cid}.attempt{attempt}.log"), "w") as lf:
            rc = self.docker(args, stdout=lf, stderr=subprocess.STDOUT).returncode
        wall = round(time.time() - t0, 1)
        self.log({"utc": start, "end": now(), "policy": pid, "cell": cid, "attempt": attempt, "exit": rc,
                  "wall_seconds": wall})
        print(f"[{self.a.stage}] {pid:12s} {cid:5s} attempt {attempt}: exit {rc}, {wall:.0f} s", flush=True)
        return rc

    def run(self):
        if self.m.get("status") != "APPROVED" or self.st.get("status") != "APPROVED":
            raise SystemExit(f"stage {self.a.stage} is not APPROVED in the manifest; refusing to run")
        cap_s = float(self.st["gpu_hour_cap"]) * 3600.0
        os.makedirs(self.out_dir, exist_ok=True)
        stopped = None
        for pid, cid in self.st["runs"]:
            out = os.path.join(self.out_dir, f"{pid}__{cid}.jsonl.gz")
            if os.path.exists(out):
                if not self.valid(out, pid, cid):
                    continue                                        # done in an earlier invocation
                self.quarantine(out, "existing output fails read_run")
            for attempt in (1, 2):
                if self.used_seconds() + PER_RUN_ESTIMATE_S > cap_s:
                    stopped = f"cap: {self.used_seconds():.0f} s used of {cap_s:.0f} s"
                    break
                rc = self.collect(pid, cid, attempt)
                if rc == 0 and os.path.exists(out) and not self.valid(out, pid, cid):
                    break
                if os.path.exists(out):
                    self.quarantine(out, f"exit {rc} or read_run problems")
                diag = out + ".diagnostic.json"
                if os.path.exists(diag):
                    self.quarantine(diag, "runtime verification failed")
            if stopped:
                print(f"== STOPPED before {pid} {cid}: {stopped}")
                break
        return self.summary(stopped)

    def summary(self, stopped):
        runs = []
        for pid, cid in self.st["runs"]:
            out = os.path.join(self.out_dir, f"{pid}__{cid}.jsonl.gz")
            probs = self.valid(out, pid, cid) if os.path.exists(out) else None
            runs.append({"policy": pid, "cell": cid,
                         "status": "missing" if probs is None else ("ok" if not probs else "invalid"),
                         "problems": probs or []})
        counts = {k: sum(1 for r in runs if r["status"] == k) for k in ("ok", "invalid", "missing")}
        rec = {"utc": now(), "stage": self.a.stage, "manifest_sha256": self.msha, "runs_total": len(runs),
               "counts": counts, "collector_wall_seconds_total": round(self.used_seconds(), 1),
               "gpu_hour_cap": self.st["gpu_hour_cap"], "stopped": stopped, "runs": runs,
               "note": "validity only; no outcomes computed"}
        with open(os.path.join(self.out_dir, f"stage_{self.a.stage}_summary.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"== {self.a.stage}: ok {counts['ok']}, invalid {counts['invalid']}, missing {counts['missing']} "
              f"of {len(runs)}; collector time {rec['collector_wall_seconds_total']:.0f} s"
              + (f"; STOPPED ({stopped})" if stopped else ""))
        return 0 if counts["ok"] == len(runs) else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--run", action="store_true")
    p.add_argument("--stage-only", action="store_true", help="preflight only the stage's policies (default: all 19)")
    p.add_argument("--host-root", default="/home/ubuntu/IsaacLab-Arena")
    p.add_argument("--container-root", default="/workspaces/isaaclab_arena")
    p.add_argument("--container", default="isaac-arena-vscode")
    p.add_argument("--docker", default="docker")
    p.add_argument("--tools-rel", default="v2_runs/phase2_eval_tools")
    a = p.parse_args(argv)
    d = Driver(a)
    return d.preflight() if a.preflight else d.run()


if __name__ == "__main__":
    sys.exit(main())
