"""v2 phase 2, part B: the study driver (PHASE2_PROTOCOL.md sections 4, 5 and 8; frozen Oct 25, 2026).

`run`:
1. checks the key holder's public key before any GPU work;
2. trains the 15 study robots (seeds 301-315, three arms) with full training records, and measures each one fully
   (nominal, check 1, check 2, Newton), qualified or not; a run failure (no final checkpoint) is retrained once;
3. re-measures the 5 controls (phase 1 seeds 101-105) under this code: nominal, check 1, check 2, Newton;
4. gives every robot a random ID, exports the feature table (rows sorted by ID; nothing identifying);
5. seals the ID -> arm/seed key and every labelled record (state, robot folders, control measurements, run folders)
   to the key holder's public key, then deletes the plain copies.

What it prints never names an arm or a seed: only counts. What stays readable afterwards: the feature table, its
SHA-256, the sealed archive and a short blind report. Resumable: finished robots are kept.
"""
import argparse
import json
import os
import shutil
import sys

from . import phase1
from .phase2_arms import task_id

STUDY_SEEDS = {"R-slip": (301, 302, 303, 304, 305), "Phi-slip-mild": (306, 307, 308, 309, 310),
               "Phi-delay": (311, 312, 313, 314, 315)}
CANDIDATE = {"R-slip": "k0.1", "Phi-slip-mild": "mild", "Phi-delay": "nodelay"}
CONTROL_SEEDS = (101, 102, 103, 104, 105)
MEASURE = ("nominal", "sweep", "check2", "newton")


def _say(msg):
    print(f"[sim2gate] {msg}", flush=True)


def _success():
    from .layer1_limits import load_success
    return load_success()


def study_plan():
    """[(name, arm, seed, task)] in seed order."""
    out = []
    for arm, seeds in STUDY_SEEDS.items():
        for s in seeds:
            out.append((f"study_s{s}", arm, s, task_id(arm, CANDIDATE[arm])))
    return sorted(out, key=lambda x: x[2])


def _run_failure(rec):
    return any(p.startswith("training exited") for p in rec.get("problems") or [])


def train_study(work, runner):
    state = phase1.load_state(work)
    plan = study_plan()
    for k, (name, arm, seed, task) in enumerate(plan, 1):
        rec = state["robots"].get(name)
        if rec and rec.get("finished") and not _run_failure(rec):
            continue
        attempts = (rec or {}).get("attempts", 0)
        while True:
            _say(f"study robot {k} of {len(plan)}: training and measuring")
            rec = phase1.train_and_measure(work, name, seed, runner, evidence=True, measure=MEASURE,
                                           require_qualify=False, task=task, success=_success())
            attempts += 1
            rec.update(arm=arm, attempts=attempts)
            state["robots"][name] = rec
            phase1.save_state(work, state)
            if not _run_failure(rec) or attempts >= 2:
                break
            _say(f"study robot {k}: run failure, retraining once from the start")
        _say(f"study robot {k} of {len(plan)}: done ({len(rec.get('problems') or [])} problems recorded)")
    return state


def remeasure_controls(work, phase1_work, runner):
    st1 = phase1.load_state(phase1_work)
    state = phase1.load_state(work)
    ctl = state.setdefault("controls", {})
    for k, seed in enumerate(CONTROL_SEEDS, 1):
        name = phase1.robot_name(seed)
        if (ctl.get(name) or {}).get("finished"):
            continue
        r = (st1.get("robots") or {}).get(name) or {}
        ck = (r.get("checkpoint") or {}).get("path")
        if not ck or not os.path.exists(ck):
            raise SystemExit(f"control {k}: checkpoint not found")
        d = os.path.join(work, "controls", name)
        os.makedirs(d, exist_ok=True)
        cmds, files = phase1.measure_commands(ck, d, task=phase1.TASK)
        _say(f"control {k} of {len(CONTROL_SEEDS)}: measuring")
        for m in MEASURE:
            runner(cmds[m], os.path.join(d, f"{m}.log"))
        ctl[name] = {"seed": seed, "measure_dir": d, "summary": phase1.summarize(files, _success()),
                     "evidence": os.path.join(phase1_work, "robots", name, "evidence"),
                     "env_yaml": os.path.join(r["run_dir"], "params", "env.yaml") if r.get("run_dir") else None,
                     "checkpoint_sha256": (r.get("checkpoint") or {}).get("sha256"), "finished": phase1._now()}
        phase1.save_state(work, state)
    return state


def export_and_seal(work, pubkey, blind_dir):
    """Random IDs, the feature table, then the sealed key and records. Returns the blind report."""
    from . import phase2_export as ex
    from ..mvp import testset
    state = phase1.load_state(work)
    robots = []
    for name, arm, seed, _ in study_plan():
        rec = state["robots"].get(name) or {}
        y = os.path.join(rec["run_dir"], "params", "env.yaml") if rec.get("run_dir") else None
        robots.append({"arm": arm, "seed": seed, "evidence": os.path.join(work, "robots", name, "evidence"),
                       "env_yaml": y, "summary": rec.get("summary"),
                       "measure": os.path.join(work, "robots", name, "measure"),
                       "run_failure": _run_failure(rec) or not rec.get("checkpoint")})
    for name, c in sorted(state.get("controls", {}).items()):
        robots.append({"arm": "Control", "seed": c["seed"], "evidence": c["evidence"], "env_yaml": c["env_yaml"],
                       "summary": c["summary"], "measure": c["measure_dir"]})
    if len(robots) != 20:
        raise SystemExit(f"expected 20 robots for export, found {len(robots)}")
    key, rows, missing_records, counts = {}, {}, 0, {"rho_undefined": 0, "rho_invalid": 0, "rho_checks": 0}
    excluded = {}
    for r in robots:
        if r.get("run_failure"):
            # Part B section 4: a run that fails again after its one retry is missing for every approach. It gets no
            # feature row; the sealed key lists it separately so the report can say so. (Added Oct 26, 2026, after the
            # phase 2 study exported seed 313's partial training record; see evidence/phase2/scoring/README.md.)
            excluded[f"x{len(excluded) + 1}"] = {"arm": r["arm"], "seed": r["seed"], "reason": "run failure twice"}
            continue
        rid = ex.new_id()
        while rid in key:
            rid = ex.new_id()
        key[rid] = {"arm": r["arm"], "seed": r["seed"]}
        if os.path.exists(os.path.join(r["evidence"], "updates.jsonl")) and r["env_yaml"] and \
                os.path.exists(r["env_yaml"]):
            f, c = ex.export_robot(r["evidence"], r["env_yaml"], r["summary"], r["measure"])
            for k in counts:
                counts[k] += c.get(k, 0)
        else:                                       # missing training record: behaviour features only
            f = {k: v for k, v in ex.behaviour_features(r["summary"], r["measure"]).items()}
            missing_records += 1
        rows[rid] = f
    os.makedirs(blind_dir, exist_ok=True)
    feat = os.path.join(blind_dir, "features.json")
    ex.write_table(rows, feat)
    keyfile = os.path.join(work, "key.json")
    with open(keyfile, "w") as f:
        json.dump({"robots": key, "excluded": excluded}, f, indent=1)
    run_dirs = sorted({rec["run_dir"] for rec in state["robots"].values() if rec.get("run_dir")})
    labelled = [keyfile, os.path.join(work, "state.json"), os.path.join(work, "robots"),
                os.path.join(work, "controls")] + [d for d in run_dirs if os.path.exists(d)]
    sealed = os.path.join(blind_dir, "sealed")
    hashes = testset.seal([p for p in labelled if os.path.exists(p)], pubkey, sealed, arc_root="/")
    for p in labelled:                                  # the plain copies go only after sealing succeeded
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)
    report = {"robots": len(rows), "excluded_run_failures": len(excluded), "features_sha256": phase1.sha256(feat), "sealed": hashes,
              "missing_training_records": missing_records, "rho": counts,
              "columns": len(json.load(open(feat))["columns"])}
    with open(os.path.join(blind_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def run(work, phase1_work, pubkey, blind_dir, runner):
    from ..mvp import testset
    _say(f"key holder's public key: {testset.check_pubkey(pubkey)}")
    train_study(work, runner)
    remeasure_controls(work, phase1_work, runner)
    rep = export_and_seal(work, pubkey, blind_dir)
    _say(f"STUDY COMPLETE: {rep['robots']} robots exported, {rep['columns']} feature columns, features sha256 "
         f"{rep['features_sha256']}; sealed {json.dumps(rep['sealed'])}; run failures excluded "
         f"{rep['excluded_run_failures']}; missing training records "
         f"{rep['missing_training_records']}; rho undefined {rep['rho']['rho_undefined']}, invalid "
         f"{rep['rho']['rho_invalid']} of {rep['rho']['rho_checks']}")
    return 0


def main(argv=None):
    from ..mvp.testset import quiet_run
    p = argparse.ArgumentParser(prog="python -m sim2gate.v2.phase2_study")
    p.add_argument("--work", required=True)
    p.add_argument("--phase1", required=True, help="phase 1's work folder (control checkpoints and records)")
    p.add_argument("--pubkey", required=True, help="the key holder's public key (vihaan_public.pem)")
    p.add_argument("--blind", required=True, help="where the feature table and the sealed archive go")
    a = p.parse_args(argv)
    return run(a.work, a.phase1, a.pubkey, a.blind, quiet_run)


if __name__ == "__main__":
    sys.exit(main())
