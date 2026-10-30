#!/usr/bin/env python3
"""Tier H (historical) comparator table for the Phase 2 development evaluation. CPU only, no simulator.

Recomputes, from the recorded Phase 2 measurement reports, each policy's:
  - nominal success under the frozen v2 episode-success rule (frozen_success_p0v5b.json);
  - Check 1 (robustness.score on sweep.json, frozen v2 success threshold);
  - Check 2 (layer1_limits.score_check2 on check2.json against the frozen combined limit), labelled with the
    task and physics it was measured under (CHECK2_PHYSICS_CONFOUND_EVIDENCE.md);
  - Check 5 (transfer.score on nominal.json and newton.json, frozen v2 success threshold);
and compares each with the summary the study driver recorded in state.json. Nothing is tuned or refit.

Usage:
    python phase2_tier_h_verdicts.py --measure-root DIR --out-json OUT.json --out-md OUT.md
DIR holds state.json, robots/study_s3NN/measure/*.json and (when available) controls/p0v5b_s10N/*.json.
"""
import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _import_checks():
    """Import the production scoring modules without Isaac Lab (same stub pattern as test_real_production_parity)."""
    if "sim2gate.checks.isaaclab_exploits" not in sys.modules:
        pkg = types.ModuleType("sim2gate")
        pkg.__path__ = [str(ROOT / "src" / "sim2gate")]
        checks = types.ModuleType("sim2gate.checks")
        checks.__path__ = [str(ROOT / "src" / "sim2gate" / "checks")]
        v2 = types.ModuleType("sim2gate.v2")
        v2.__path__ = [str(ROOT / "src" / "sim2gate" / "v2")]
        meas = types.ModuleType("sim2gate.checks.isaaclab_exploits")
        meas.CALIBRATED_METRICS = ("foot_slip_p95_mps", "ground_penetration_m", "contact_force_p99_bw",
                                   "action_jitter_high_freq_power_share", "actuator_limit_step_share")
        meas.MEASUREMENT_KEYS = ("dt", "contact_force_n", "foot_radius_m", "penetration_tol_m", "jitter_window",
                                 "flat_ground", "measurement_version", "foot_geometry")
        sys.modules.update({"sim2gate": pkg, "sim2gate.checks": checks, "sim2gate.v2": v2,
                            "sim2gate.checks.isaaclab_exploits": meas})
    from sim2gate.checks import robustness, transfer
    from sim2gate.v2 import layer1_limits
    return robustness, transfer, layer1_limits


ARMS = {"R-slip": "R-slip", "Phi-slip-mild": "Φ-slip-mild", "Phi-delay": "Φ-delay"}
PHYSICS = {
    "Sim2Gate-Velocity-Flat-Go2-P0-v5b": "P0 v5b: static friction 0.35-1.2, delay 0-30 ms",
    "Sim2Gate-Velocity-Flat-Go2-P2-R-slip-k0.1": "P0 v5b physics (reward change only)",
    "Sim2Gate-Velocity-Flat-Go2-P2-Phi-slip-mild-mild": "TRAINING PHYSICS: static friction 0.3-0.6",
    "Sim2Gate-Velocity-Flat-Go2-P2-Phi-delay-nodelay": "TRAINING PHYSICS: no command delay",
}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _task(report):
    a = report.get("play_args") or []
    return next((a[i + 1] for i, x in enumerate(a[:-1]) if x == "--task"), report.get("task"))


def score_policy(name, arm, folder, recorded, success, limit_doc, mods):
    robustness, transfer, layer1_limits = mods
    files = {m: os.path.join(folder, f"{m}.json") for m in ("nominal", "sweep", "check2", "newton")}
    row = {"policy_id": name, "arm": arm, "folder": folder, "inputs": {}}
    missing = [m for m, p in files.items() if not os.path.exists(p)]
    if missing:
        row["status"] = f"missing reports: {', '.join(missing)}"
        return row
    raw = {m: json.load(open(p)) for m, p in files.items()}
    row["inputs"] = {m: _sha(p) for m, p in files.items()}
    thr = success["threshold_mps"]
    nom_eps = raw["nominal"]["episodes"]
    row["nominal_success"] = {"rate": robustness.success_rate(nom_eps, thr), "episodes": len(nom_eps),
                              "falls": sum(bool(e["fell"]) for e in nom_eps)}
    c1 = robustness.score(raw["sweep"], success)
    row["check1"] = {"status": c1.get("status"), "flagged": c1.get("flagged"),
                     "flagged_params": sorted(k for k, v in (c1.get("params") or {}).items() if v.get("flagged")),
                     "reasons": c1.get("reasons")}
    c2 = layer1_limits.score_check2(raw["check2"], limit_doc)
    task = _task(raw["check2"])
    row["check2"] = {"status": c2.get("status"), "flagged": c2.get("flagged"), "score": c2.get("score"),
                     "limit": c2.get("limit"), "driving_metric": c2.get("driving_metric"),
                     "reasons": c2.get("reasons"), "measured_in_task": task,
                     "physics_label": PHYSICS.get(task, "UNKNOWN TASK")}
    c5 = transfer.score(raw["nominal"], raw["newton"], success)
    row["check5"] = {"status": c5.get("status"), "flagged": c5.get("flagged"), "reasons": c5.get("reasons"),
                     "risks": sorted(r.get("metric") for r in (c5.get("risks") or []))}
    flags = [row[c]["flagged"] for c in ("check1", "check2", "check5")]
    row["sim2gate_any_flag"] = None if any(f is None for f in flags) else any(flags)
    rs = recorded or {}
    row["recorded"] = {"nominal_success": (rs.get("nominal") or {}).get("success_rate"),
                       "check1_flagged": (rs.get("check1") or {}).get("flagged")}
    row["recomputed_matches_recorded"] = {
        "nominal_success": row["recorded"]["nominal_success"] == row["nominal_success"]["rate"],
        "check1_flagged": row["recorded"]["check1_flagged"] == row["check1"]["flagged"]}
    row["status"] = "scored"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md")
    a = ap.parse_args(argv)
    mods = _import_checks()
    layer1_limits = mods[2]
    success, limit_doc = layer1_limits.load_success(), layer1_limits.load_limit()
    root = Path(a.measure_root)
    state = json.load(open(root / "state.json"))
    rows = []
    for name, c in sorted(state["controls"].items()):
        rows.append(score_policy(name, "control", str(root / "controls" / name), c.get("summary"),
                                 success, limit_doc, mods))
    for name, r in sorted(state["robots"].items(), key=lambda kv: kv[1]["seed"]):
        pid = f"s{r['seed']}"
        arm = ARMS.get(r.get("arm"), r.get("arm"))
        if not (r.get("checkpoint") or {}).get("sha256"):
            rows.append({"policy_id": pid, "arm": arm, "status": "missing: no final checkpoint (incomplete run)"})
            continue
        rows.append(score_policy(pid, arm, str(root / "robots" / name / "measure"), r.get("summary"),
                                 success, limit_doc, mods))
    out = {"tier": "H (historical, recorded Phase 2 measurements; development only)",
           "rules": {"success_threshold_mps": success["threshold_mps"], "success_source": "src/sim2gate/v2/frozen_success_p0v5b.json",
                     "check2_limit": limit_doc["limit"], "check2_source": "src/sim2gate/v2/check2_limit_v2.json",
                     "check1": "sim2gate.checks.robustness.score, v2 envelope", "check5": "sim2gate.checks.transfer.score"},
           "tool_sha256": _sha(__file__), "state_sha256": _sha(root / "state.json"), "rows": rows}
    Path(a.out_json).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    lines = ["| policy | arm | nominal success | Check 1 | Check 2 (score / limit) | Check 2 measured under | Check 5 | any Sim2Gate flag |",
             "|---|---|---|---|---|---|---|---|"]
    fmt = lambda f: "—" if f is None else ("FLAG" if f else "pass")
    for r in rows:
        if r["status"] != "scored":
            lines.append(f"| {r['policy_id']} | {r['arm']} | {r['status']} |  |  |  |  |  |")
            continue
        ns, c1, c2, c5 = r["nominal_success"], r["check1"], r["check2"], r["check5"]
        c1s = fmt(c1["flagged"]) + (f" ({', '.join(c1['flagged_params'])})" if c1["flagged_params"] else "")
        c2s = (f"{fmt(c2['flagged'])} ({c2['score']:.3f} / {c2['limit']:.4f}, {c2['driving_metric']})"
               if c2["status"] == "scored" else f"invalid: {'; '.join(c2['reasons'])[:60]}")
        c5s = fmt(c5["flagged"]) + (f" ({', '.join(c5['risks'])})" if c5["risks"] else "")
        lines.append(f"| {r['policy_id']} | {r['arm']} | {ns['rate']:.3f} ({ns['episodes']} ep, {ns['falls']} falls) | "
                     f"{c1s} | {c2s} | {c2['physics_label']} | {c5s} | {fmt(r['sim2gate_any_flag'])} |")
    if a.out_md:
        Path(a.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
