"""v2 phase 1: the clean robots (docs/v2/PHASE1_PROTOCOL.md, frozen before any phase 1 training).

    pilot   seeds 1-2 of the current P0 (TASK: P0 v5b since Oct 21, 2026; P0 v5's pilot failed): train, qualify,
            measure, and apply the frozen pilot rule. The recipe is frozen only if the pilot passes; otherwise it is
            revised as a new version and piloted again.
    clean   the 20 calibration robots, seeds 101-120 (replacement seeds 121-125 if a robot fails qualification),
            trained with the full training record (EvidenceRecorder) and measured for Layer 1's limits.
    bundle  one .tgz of everything (records, logs, measurements, checkpoints) with a SHA-256 manifest.

Every robot is measured the same way: nominal PhysX 256 episodes (qualification), check 1 swept over the v2
envelope, check 2, and nominal Newton 256 episodes (check 5). Nothing is blinded: these robots are all clean.

Inside the Isaac Lab container, from the Isaac Lab-Arena folder:
    python -m sim2gate.v2.phase1 smoke  --work v2_runs/phase1_smoke_v5b     # ~20 min: every new piece once
    python -m sim2gate.v2.phase1 pilot  --work v2_runs/phase1_pilot_v5b
    python -m sim2gate.v2.phase1 clean  --work v2_runs/phase1_clean
    python -m sim2gate.v2.phase1 bundle --work v2_runs/phase1_clean --out phase1-clean.tgz
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
import tarfile

from .tasks import CALLBACK, P0_TASK_CURRENT, P0_V5_ITERATIONS, P0_V5B_PAYLOAD_RAMP_ROUNDS, ROUND_ENV_STEPS, VARIANTS

TASK = P0_TASK_CURRENT                   # P0 v5b; P0 v5's pilot (Oct 21, 2026) failed, see PHASE1_FREEZE.md
LABEL = f"P0 v{VARIANTS[TASK]['version']}"
PREFIX = f"p0v{VARIANTS[TASK]['version']}"


def robot_name(seed):
    return f"{PREFIX}_s{seed}"

PILOT_SEEDS = (1, 2)
CLEAN_SEEDS = tuple(range(101, 121))
REPLACEMENT_SEEDS = tuple(range(121, 126))
QUALIFY_MIN = 0.85                       # v2 plan: qualify at 85% or more over 256 episodes
REPEAT = 4                               # 4 x the 64-command set = 256 episodes
ENVELOPE = "v2"
PROVISIONAL_SUCCESS = "frozen_success_p0v4.json"   # v1's success threshold, until the 20 clean robots recalibrate it
MEASUREMENTS = ("nominal", "sweep", "check2", "newton")


# ---- commands ------------------------------------------------------------------------------------------------------

def train_command(seed, work, name, evidence=True, iterations=P0_V5_ITERATIONS):
    from ..checks import transfer
    from ..training.smoke_go2 import isaac_python
    d = os.path.join(work, "robots", name)
    cmd = isaac_python() + ["-m", "sim2gate.training.isaaclab_launch", "--eta", "0",
                            "--signals", os.path.join(d, "monitor.jsonl")]
    if evidence:
        cmd += ["--evidence", os.path.join(d, "evidence")]
    return cmd + ["--", "--task", TASK, "--seed", str(seed), "--max_iterations", str(iterations)] \
        + CALLBACK + [transfer.PHYSX]


def measure_commands(ckpt, out_dir):
    from ..checks import transfer
    from ..mvp.baselines import GO2_COMMANDS
    from ..mvp.blinded import CHECK2
    from ..training.smoke_go2 import isaac_python
    py = isaac_python()
    task = ["--task", TASK]
    fixed = list(transfer.FIXED_OVERRIDES) + list(transfer.ORDERING)
    o = {m: os.path.join(out_dir, f"{m}.json") for m in MEASUREMENTS}

    def cond(mode, physics, out, repeat):
        return py + ["-m", "sim2gate.checks.isaaclab_conditions", "--mode", mode, "--envelope", ENVELOPE,
                     "--repeat", str(repeat), "--commands", GO2_COMMANDS, "--out", out, "--"] + task \
            + ["--checkpoint", ckpt] + CALLBACK + [physics] + fixed
    return {
        "nominal": cond("nominal", transfer.PHYSX, o["nominal"], REPEAT),
        "sweep": cond("sweep", transfer.PHYSX, o["sweep"], 1),
        "check2": py + ["-m", "sim2gate.checks.isaaclab_exploits", "--steps", str(CHECK2["steps"]), "--commands",
                        GO2_COMMANDS, "--out", o["check2"], "--"] + task
        + ["--num_envs", str(CHECK2["num_envs"]), "--checkpoint", ckpt] + CALLBACK + [CHECK2["physics"]],
        "newton": cond("nominal", transfer.NEWTON, o["newton"], REPEAT),
    }, o


# ---- helpers -------------------------------------------------------------------------------------------------------

def _say(msg):
    print(f"[sim2gate] {msg}", flush=True)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_state(work):
    p = os.path.join(work, "state.json")
    return json.load(open(p)) if os.path.exists(p) else {"created": _now(), "robots": {}}


def save_state(work, state):
    p = os.path.join(work, "state.json")
    with open(p + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(p + ".tmp", p)


def success_doc():
    from ..mvp import blinded
    return json.load(open(os.path.join(os.path.dirname(blinded.__file__), PROVISIONAL_SUCCESS)))


def summarize(files, success):
    """Qualification and the check results of one robot from its measurement files."""
    from ..checks import robustness
    out = {}
    if os.path.exists(files["nominal"]):
        raw = json.load(open(files["nominal"]))
        rate = robustness.success_rate(raw["episodes"], success["threshold_mps"])
        out["nominal"] = {"success_rate": rate, "episodes": len(raw["episodes"]),
                          "falls": sum(bool(e["fell"]) for e in raw["episodes"]),
                          "qualifies": rate is not None and rate >= QUALIFY_MIN,
                          "checkpoint_sha256": raw.get("checkpoint_sha256")}
    if os.path.exists(files["sweep"]):
        res = robustness.score(json.load(open(files["sweep"])), success)
        out["check1"] = {"status": res["status"], "envelope": res.get("envelope"), "flagged": res["flagged"],
                         "reasons": res["reasons"],
                         "cliffs": {k: v["cliff"] for k, v in res["params"].items()},
                         "flagged_params": [k for k, v in res["params"].items() if v["flagged"]]}
    if os.path.exists(files["check2"]):
        out["check2_metrics"] = json.load(open(files["check2"])).get("metrics")
    if os.path.exists(files["newton"]):
        raw = json.load(open(files["newton"]))
        out["newton"] = {"success_rate": robustness.success_rate(raw["episodes"], success["threshold_mps"]),
                         "falls": sum(bool(e["fell"]) for e in raw["episodes"]), "episodes": len(raw["episodes"])}
    return out


def train_and_measure(work, name, seed, runner, evidence, measure=MEASUREMENTS, iterations=P0_V5_ITERATIONS,
                      require_qualify=True):
    from ..mvp.baselines import shutdown_crash
    from ..training.smoke_go2 import _last_checkpoint, _run_dirs
    d = os.path.join(work, "robots", name)
    if os.path.exists(d):
        os.replace(d, d + f".incomplete-{datetime.datetime.now():%Y%m%d%H%M%S}")
    os.makedirs(d)
    rec = {"name": name, "seed": seed, "task": TASK, "started": _now(), "problems": []}
    before = _run_dirs()
    rc, text = runner(train_command(seed, work, name, evidence, iterations), os.path.join(d, "train.log"))
    new = sorted(_run_dirs() - before)
    ck = _last_checkpoint(new[0]) if len(new) == 1 else None
    rec.update(train_returncode=rc, run_dir=os.path.abspath(new[0]) if new else None, trained=_now())
    crash = shutdown_crash(rc, text, ck, iterations) if ck else None
    if crash:
        rec["train_shutdown_crash"] = crash
    err = os.path.join(d, "evidence", "error.txt")
    if os.path.exists(err):
        lines = [x for x in open(err).read().splitlines() if x.strip()]
        rec["problems"].append("recorder error: " + " | ".join(lines[:1] + lines[-1:]))
    if ck is None or (rc != 0 and not crash):
        rec["problems"].append(f"training exited {rc}; {len(new)} new run folder(s)")
        return rec
    rec["checkpoint"] = {"path": os.path.abspath(ck), "sha256": sha256(ck)}
    out = os.path.join(d, "measure")
    os.makedirs(out, exist_ok=True)
    cmds, files = measure_commands(rec["checkpoint"]["path"], out)
    success = success_doc()
    for m in measure:
        runner(cmds[m], os.path.join(out, f"{m}.log"))
        if not os.path.exists(files[m]):
            rec["problems"].append(f"{m}: no output")
        rec["summary"] = summarize(files, success)
        nom = rec["summary"].get("nominal")
        if require_qualify and m == "nominal" and (nom is None or not nom["qualifies"]):
            rec["summary"]["not_measured_further"] = "did not qualify"
            break                                   # a robot that does not qualify is not measured further
    nom = rec.get("summary", {}).get("nominal")
    if nom and nom.get("checkpoint_sha256") != rec["checkpoint"]["sha256"]:
        rec["problems"].append("the measured checkpoint is not the trained one")
    rec["finished"] = _now()
    return rec


# ---- smoke ---------------------------------------------------------------------------------------------------------

def smoke_verdict(rec):
    """A short run (seed 99, 10 rounds, full training record) that exercises every new piece once."""
    problems = list(rec.get("problems") or [])
    d = rec.get("_measure_dir")
    sw = os.path.join(d, "sweep.json") if d else None
    if not (sw and os.path.exists(sw)):
        problems.append("no check 1 sweep output")
    else:
        raw = json.load(open(sw))
        if raw.get("envelope") != ENVELOPE:
            problems.append(f"sweep envelope {raw.get('envelope')!r}, expected {ENVELOPE!r}")
        tr = raw.get("training_ranges") or {}
        pay = (tr.get("added_base_mass_kg") or {}).get("payload") if isinstance(tr.get("added_base_mass_kg"),
                                                                                dict) else None
        if not pay or list(pay.get("params") or []) != [0.0, 8.0]:
            problems.append(f"training payload not recorded as 0-8 kg: {tr.get('added_base_mass_kg')}")
        elif VARIANTS[TASK]["version"] == "5b":
            ramp = [r * ROUND_ENV_STEPS for r in P0_V5B_PAYLOAD_RAMP_ROUNDS]
            if pay.get("mode") != "reset" or list(pay.get("ramp_env_steps") or []) != ramp:
                problems.append(f"training payload not per episode on the ramp {ramp}: {pay}")
        ms = tr.get("motor_strength_scale")
        if not (isinstance(ms, dict) and any(list(v) == [0.8, 1.0] for v in ms.values())):
            problems.append(f"training motor strength not recorded as 0.8-1.0: {ms}")
        if "add_payload" not in ((raw.get("profile_record") or {}).get("removed_events") or []):
            problems.append("the nominal profile did not remove the training payload")
    s = rec.get("summary") or {}
    for m in MEASUREMENTS:
        key = {"nominal": "nominal", "sweep": "check1", "check2": "check2_metrics", "newton": "newton"}[m]
        if s.get(key) is None:
            problems.append(f"{m}: no result")
    return {"pass": not problems, "problems": problems}


def smoke(work, runner, iterations=10):
    os.makedirs(work, exist_ok=True)
    rec = train_and_measure(work, "smoke_s99", 99, runner, evidence=True, iterations=iterations,
                            require_qualify=False)
    rec["_measure_dir"] = os.path.join(work, "robots", "smoke_s99", "measure")
    v = smoke_verdict(rec)
    json.dump({"robot": rec, "verdict": v}, open(os.path.join(work, "smoke.json"), "w"), indent=1)
    s = rec.get("summary") or {}
    _say(f"nominal {json.dumps(s.get('nominal'))}")
    _say(f"check 1 {json.dumps(s.get('check1'))}")
    _say("SMOKE OK" if v["pass"] else "SMOKE FAILED: " + " | ".join(v["problems"]))
    return 0 if v["pass"] else 1


# ---- pilot ---------------------------------------------------------------------------------------------------------

def pilot_verdict(robots):
    """The frozen pilot rule: every pilot seed qualifies (>= 85% of 256 nominal episodes) and has no check-1 flag
    inside the v2 envelope (a scored sweep, no parameter's cliff inside its plausible range)."""
    reasons = []
    for name, r in sorted(robots.items()):
        s = r.get("summary") or {}
        nom, c1 = s.get("nominal"), s.get("check1")
        if r.get("problems"):
            reasons.append(f"{name}: {'; '.join(r['problems'])}")
        if not nom:
            reasons.append(f"{name}: no qualification result")
        elif not nom["qualifies"]:
            reasons.append(f"{name}: nominal success {nom['success_rate']:.3f} < {QUALIFY_MIN} over "
                           f"{nom['episodes']} episodes")
        if not c1:
            reasons.append(f"{name}: no check 1 result")
        elif c1["status"] != "scored":
            reasons.append(f"{name}: check 1 {c1['status']}: {'; '.join(c1['reasons'])}")
        elif c1["flagged"]:
            reasons.append(f"{name}: check 1 flags {', '.join(c1['flagged_params'])} inside the v2 envelope "
                           f"(cliffs {c1['cliffs']})")
    return {"pass": not reasons and len(robots) == len(PILOT_SEEDS), "reasons": reasons}


def pilot(work, runner):
    os.makedirs(work, exist_ok=True)
    state = load_state(work)
    for seed in PILOT_SEEDS:
        name = robot_name(seed)
        if (state["robots"].get(name) or {}).get("finished") and not state["robots"][name].get("problems"):
            continue
        _say(f"pilot: {name} (training, then measuring)")
        state["robots"][name] = train_and_measure(work, name, seed, runner, evidence=False)
        save_state(work, state)
        _say(f"  {name}: {json.dumps(state['robots'][name].get('summary', {}).get('nominal'))}")
    v = pilot_verdict(state["robots"])
    state["pilot_verdict"] = v
    save_state(work, state)
    for name, r in sorted(state["robots"].items()):
        s = r.get("summary") or {}
        _say(f"{name}: nominal {json.dumps(s.get('nominal'))}")
        _say(f"{name}: check 1 {json.dumps(s.get('check1'))}")
        _say(f"{name}: check 2 {json.dumps(s.get('check2_metrics'))}")
        _say(f"{name}: newton {json.dumps(s.get('newton'))}")
    _say(f"PILOT PASS: {LABEL} can be frozen" if v["pass"] else "PILOT FAIL: " + " | ".join(v["reasons"]))
    return 0 if v["pass"] else 1


# ---- the 20 clean robots -------------------------------------------------------------------------------------------

def clean(work, runner, pilot_work=None):
    if pilot_work:
        ps = load_state(pilot_work)
        if not (ps.get("pilot_verdict") or {}).get("pass"):
            raise SystemExit(f"the pilot in {pilot_work} has not passed; {LABEL} is not frozen")
    os.makedirs(work, exist_ok=True)
    state = load_state(work)
    state.setdefault("plan", {"seeds": list(CLEAN_SEEDS), "replacements": list(REPLACEMENT_SEEDS),
                              "qualify_min": QUALIFY_MIN, "episodes": 64 * REPEAT, "envelope": ENVELOPE})
    queue = list(CLEAN_SEEDS)
    spare = list(REPLACEMENT_SEEDS)
    qualified = []
    while queue:
        seed = queue.pop(0)
        name = robot_name(seed)
        r = state["robots"].get(name)
        if not (r and r.get("finished") and not r.get("problems")):
            _say(f"clean robot {name} ({len(qualified) + 1} of {len(CLEAN_SEEDS)} needed)")
            r = train_and_measure(work, name, seed, runner, evidence=True)
            state["robots"][name] = r
            save_state(work, state)
        nom = (r.get("summary") or {}).get("nominal") or {}
        if r.get("problems") or not nom.get("qualifies"):
            _say(f"  {name} not used: {'; '.join(r.get('problems') or [])} "
                 f"nominal {nom.get('success_rate')}")
            if not spare:
                state["result"] = "stopped: replacement seeds used up"
                save_state(work, state)
                _say("STOPPED: more than 5 robots failed; decision needed")
                return 1
            queue.append(spare.pop(0))
            continue
        qualified.append(name)
        _say(f"  {name} qualified ({nom['success_rate']:.3f}); check 1 flags: "
             f"{r['summary'].get('check1', {}).get('flagged_params')}")
    state["calibration_set"] = qualified
    state["result"] = "complete"
    save_state(work, state)
    _say(f"CLEAN ROBOTS COMPLETE: {len(qualified)} qualified")
    return 0


# ---- bundle --------------------------------------------------------------------------------------------------------

def bundle(work, out):
    state = load_state(work)
    files = []
    for name, r in sorted(state["robots"].items()):
        d = os.path.join(work, "robots", name)
        for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if os.path.isfile(p):
                files.append((p, os.path.relpath(p, work)))
        ck = (r.get("checkpoint") or {}).get("path")
        if ck and os.path.exists(ck):
            files.append((ck, os.path.join("robots", name, "checkpoint", os.path.basename(ck))))
            params = os.path.join(os.path.dirname(ck), "params")
            for p in sorted(glob.glob(os.path.join(params, "*"))):
                files.append((p, os.path.join("robots", name, "checkpoint", "params", os.path.basename(p))))
    manifest = {"created": _now(), "state": state, "files": {arc: sha256(p) for p, arc in files}}
    mp = os.path.join(work, "MANIFEST.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=1)
    base = os.path.basename(os.path.normpath(work))
    with tarfile.open(out, "w:gz") as t:
        t.add(mp, arcname=f"{base}/MANIFEST.json")
        for p, arc in files:
            t.add(p, arcname=os.path.join(base, arc))
    _say(f"bundle {out}: {len(files)} files, {os.path.getsize(out) / 1e6:.0f} MB, sha256 {sha256(out)}")
    return 0


def main(argv=None):
    from ..mvp.testset import quiet_run
    p = argparse.ArgumentParser(prog="python -m sim2gate.v2.phase1", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("smoke", "pilot", "clean", "bundle"):
        s = sub.add_parser(c)
        s.add_argument("--work", required=True)
        if c == "clean":
            s.add_argument("--pilot", default=None, help="the pilot's work folder: refuse to start unless it passed")
        if c == "bundle":
            s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "smoke":
        return smoke(a.work, quiet_run)
    if a.cmd == "pilot":
        return pilot(a.work, quiet_run)
    if a.cmd == "clean":
        return clean(a.work, quiet_run, a.pilot)
    return bundle(a.work, a.out)


if __name__ == "__main__":
    sys.exit(main())
