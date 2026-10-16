"""Run checks 1 (robustness sweep) and 5 (PhysX vs Newton) on trained policies, and score them.

MVP plan, step 2, for checks 1 and 5: on the calibration baselines (P0 seeds 1-3) only. Run inside the Isaac Lab
container, from the Isaac Lab-Arena folder, after the P0 baselines (sim2gate.mvp.baselines) are trained:

    python -m sim2gate.mvp.checks15 --baselines mvp_runs/P0 --seeds 1 2 3 --out mvp_runs/P0_checks15 --calibrate

For each seed it takes the checkpoint recorded in the baselines manifest and runs three measurements
(sim2gate.checks.isaaclab_conditions), each a separate Isaac Sim launch:
  sweep   check 1's grid on PhysX: 46 condition groups x 64 episodes (2,944 environments)
  physx   check 5's nominal run on PhysX (64 episodes)
  newton  check 5's nominal run on Newton / MuJoCo-Warp (64 episodes)
With --calibrate it then sets the episode-success reference from the three sweeps' nominal episodes
(sim2gate.checks.robustness.success_reference; P0 seeds 1-3 only, never overwritten), and scores check 1 and
check 5 for every seed with it. Every raw file is checked against the checkpoint hash in the manifest.

A run counts by its output, not its exit code: Isaac Sim sometimes crashes while shutting down after the output is
written (seen in training, C5); the exit code is recorded either way.

--score-only launches nothing: it re-checks the raw files already in --out against the manifest's checkpoints and
scores them (plan change C8, after the success reference was refused on Oct 15).
"""
import argparse
import json
import os
import sys

from ..checks import robustness, transfer
from ..training.smoke_go2 import isaac_python, run
from .baselines import GO2_COMMANDS, _callback, sha256
from .tasks import P0_TASK

FIXED = list(transfer.FIXED_OVERRIDES) + list(transfer.ORDERING)
RUNS = {"sweep": ("sweep", "physics=isaacsim_physx"), "physx": ("nominal", "physics=isaacsim_physx"),
        "newton": ("nominal", "physics=newton_mjwarp")}


def run_cmd(a, mode, physics, ckpt, out):
    return isaac_python() + ["-m", "sim2gate.checks.isaaclab_conditions", "--mode", mode, "--commands", a.commands,
                             "--out", out, "--", "--task", a.task, "--checkpoint", ckpt] \
        + _callback(a) + [physics] + FIXED


def _save(doc, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.mvp.checks15", description=__doc__.splitlines()[0])
    p.add_argument("--baselines", required=True, help="the baselines folder (its manifest.json names the checkpoints)")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task", default=P0_TASK)
    p.add_argument("--commands", default=GO2_COMMANDS)
    p.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS))
    p.add_argument("--calibrate", action="store_true",
                   help="set the episode-success reference from these policies (P0 seeds 1-3 only)")
    p.add_argument("--success", default=None, help="an existing success reference (success.json) to score with")
    p.add_argument("--score-only", action="store_true",
                   help="launch nothing; score the raw files already in --out (each re-checked against its checkpoint)")
    a = p.parse_args(argv)
    if a.calibrate and sorted(a.seeds) != [1, 2, 3]:
        p.error("the episode-success reference is set from P0 seeds 1 to 3 only")
    if a.calibrate and a.success:
        p.error("--calibrate sets the success reference; do not pass --success as well")
    bm = json.load(open(os.path.join(a.baselines, "manifest.json")))
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "manifest.json")
    m = json.load(open(mpath)) if os.path.exists(mpath) else {}
    m.update({"baselines": os.path.abspath(a.baselines), "task": a.task, "fixed_overrides": FIXED,
              "grid": robustness.grid_definition(), "seeds": m.get("seeds", {})})
    ok = True
    for s in a.seeds:
        e = (bm.get("seeds") or {}).get(str(s)) or {}
        ck = (e.get("checkpoint") or {})
        if not ck.get("path") or not os.path.exists(ck["path"]) or sha256(ck["path"]) != ck.get("sha256"):
            print(f"[sim2gate] seed {s}: checkpoint missing or changed since the baselines manifest", file=sys.stderr)
            m["seeds"][str(s)] = {"error": "checkpoint missing or changed"}
            ok = False
            continue
        rec = m["seeds"].setdefault(str(s), {})
        rec["checkpoint"] = ck
        for name in a.runs:
            mode, physics = RUNS[name]
            out = os.path.join(a.out, f"raw_{name}_s{s}.json")
            if a.score_only:
                r = dict(rec.get(name) or {})
                r.pop("error", None)
                r.pop("error_file", None)
            else:
                for f in (out, out + ".error.txt"):
                    if os.path.exists(f):
                        os.remove(f)
                rc, _ = run(run_cmd(a, mode, physics, ck["path"], out), os.path.join(a.out, f"{name}_s{s}.log"))
                r = {"returncode": rc}
            if not os.path.exists(out):
                r["error"] = f"no output (exit {r.get('returncode')})"
                if os.path.exists(out + ".error.txt"):
                    lines = [x for x in open(out + ".error.txt").read().splitlines() if x.strip()]
                    r["error"] += ": " + (lines[-1] if lines else "")
                    r["error_file"] = os.path.abspath(out + ".error.txt")
                ok = False
            else:
                raw = json.load(open(out))
                r["raw"] = {"path": os.path.abspath(out), "sha256": sha256(out)}
                if raw.get("checkpoint_sha256") != ck["sha256"]:
                    r["error"] = f"measured checkpoint {raw.get('checkpoint_sha256')}, not {ck['sha256']}"
                    ok = False
                elif not raw.get("all_ended"):
                    r["error"] = "not every episode ended"
                    ok = False
            rec[name] = r
            _save(m, mpath)
    succ_path = a.success or os.path.join(a.out, "success.json")
    if a.calibrate and ok:
        if os.path.exists(succ_path):
            print(f"[sim2gate] {succ_path} exists; not overwriting the success reference", file=sys.stderr)
            ok = False
        else:
            try:
                doc = robustness.success_reference(
                    [(f"P0 seed {s}", json.load(open(m["seeds"][str(s)]["sweep"]["raw"]["path"]))) for s in a.seeds])
            except (robustness.SuccessError, KeyError) as err:
                print(f"[sim2gate] success reference refused: {err}", file=sys.stderr)
                m["success"] = {"error": str(err)}
                ok = False
            else:
                _save(doc, succ_path)
                m["success"] = {"path": os.path.abspath(succ_path), "sha256": sha256(succ_path),
                                "threshold_mps": doc["threshold_mps"]}
    succ = robustness.load_success(succ_path) if os.path.exists(succ_path) else None
    print("\n" + "=" * 80)
    if succ:
        print(f"episode success: no fall and mean tracking error <= {succ['threshold_mps']:.4f} m/s "
              f"({succ['factor']} x {succ['reference_error_mps']:.4f})")
    for s in a.seeds:
        rec = m["seeds"].get(str(s), {})
        line = f"P0 seed {s}:"
        if succ and "raw" in rec.get("sweep", {}):
            c1 = robustness.score(json.load(open(rec["sweep"]["raw"]["path"])), succ)
            _save(c1, os.path.join(a.out, f"check1_s{s}.json"))
            rec["check1"] = {"status": c1["status"], "flagged": c1["flagged"], "reasons": c1["reasons"]}
            cl = ", ".join(f"{k} {v['cliff'] if v['cliff'] is not None else '-'}{' FLAG' if v['flagged'] else ''}"
                           for k, v in c1["params"].items())
            line += f"\n  check 1: {c1['status']}, nominal success {c1['nominal_success_rate']}, cliffs: {cl}"
        if succ and "raw" in rec.get("physx", {}) and "raw" in rec.get("newton", {}):
            c5 = transfer.score(json.load(open(rec["physx"]["raw"]["path"])),
                                json.load(open(rec["newton"]["raw"]["path"])), succ)
            _save(c5, os.path.join(a.out, f"check5_s{s}.json"))
            rec["check5"] = {"status": c5["status"], "flagged": c5["flagged"], "reasons": c5["reasons"]}
            mt = c5["metrics"]
            line += (f"\n  check 5: {c5['status']} (parity {c5['parity']}), success PhysX {mt['physx_success_rate']} "
                     f"Newton {mt['newton_success_rate']}, Newton fall-free {mt['newton_fall_free_rate']}, error ratio "
                     f"{'n/a' if mt['error_ratio'] is None else round(mt['error_ratio'], 3)}"
                     f"{', FLAGGED' if c5['flagged'] else ''}")
        for name in a.runs:
            if (rec.get(name) or {}).get("error"):
                line += f"\n  {name}: ERROR {rec[name]['error']}"
        print(line)
        for k in ("check1", "check5"):
            for r in (rec.get(k) or {}).get("reasons") or []:
                print(f"    {k} invalid: {r}")
    if a.score_only:
        m["scored_only"] = True
    _save(m, mpath)
    print(f"\n{'CHECKS 1 AND 5 DONE' if ok else 'CHECKS 1 AND 5 INCOMPLETE'}  (manifest: {mpath})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
