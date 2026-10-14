"""Train a set of MVP policies and run check 2 on each; for the calibration baselines, set the thresholds.

MVP plan, step 2: "Build the checks and set thresholds using only P0 seeds 1 to 3". Run inside the Isaac Lab
container, from the Isaac Lab-Arena folder:

    python -m sim2gate.mvp.baselines --policy P0 --seeds 1 2 3 --out mvp_runs/P0 --calibrate

For each seed it trains Isaac Lab's Go2 flat task with stock RSL-RL PPO on Isaac Sim's PhysX (physics=isaacsim_physx, the plan's
training engine), finds
the final checkpoint, and runs check 2 on it under the same physics with the fixed command set (64 envs, 1000
steps; the plan's set ships with Sim2Gate: 64 commands, seed 20261012, see GO2_COMMANDS). Every checkpoint is recorded with its SHA-256, and the run's saved env and agent configs with theirs.
With --calibrate it then sets the thresholds from the check 2 reports: max over the baselines x1.5, as the plan
says (sim2gate.checks.calibrate).

Everything goes into --out: manifest.json (written after every step; on a rerun, seeds whose checkpoint still
matches its recorded hash are not retrained), one train and one check 2 log per seed, the check 2 reports, and
thresholds.json. Exit code 0 only if every seed trained, every check 2 run produced a report and, with
--calibrate, the thresholds were written.

Only the planted-flaw configuration changes, given with --train-arg (Hydra overrides such as
env.rewards.action_rate_l2.weight=0.0), separate one policy set from another; they are recorded in the manifest.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys

from ..training.smoke_go2 import _last_checkpoint, _run_dirs, isaac_python, run, train_script
from .tasks import P0_TASK, VARIANTS

# Acceptance criteria for a calibration baseline (plan change C4), fixed before the P0 v1 baselines were trained.
# Measured by check 2 on the fixed command set (64 envs, 1000 steps = one 20 s episode each). A baseline that misses
# any of them is not used for calibration.
BASELINE_CRITERIA = {
    # command tracking: mean |commanded - actual| base xy velocity. 0.15 m/s is about 1.5x the 0.09-0.10 m/s of the
    # stock P0 policies (Oct 13), the plan's own tolerance for episode success
    "max_mean_tracking_error_mps": 0.15,
    # falls: episodes ended before the time limit, at most 1 of the 64 (the plan's episode success needs no fall)
    "max_falls": 1,
    # sustained-contact slip: p95 horizontal speed of feet in contact for >= 4 physics updates in a row. 0.5 m/s is
    # about half the median commanded speed (0.9 m/s): a planted foot moving that fast is sliding, not rolling
    # (the 22 mm foot sphere pivoting on the calf moves its centre by at most ~0.2-0.4 m/s)
    "max_sustained_slip_p95_mps": 0.5,
}


def baseline_acceptance(rep):
    """(accepted, details) for one check 2 report against BASELINE_CRITERIA."""
    nm, mt = rep.get("nominal") or {}, rep.get("metrics") or {}
    got = {"mean_tracking_error_mps": nm.get("mean_lin_vel_error_xy_mps"), "falls": nm.get("falls"),
           "sustained_slip_p95_mps": mt.get("foot_slip_sustained_p95_mps")}
    c = BASELINE_CRITERIA
    checks = {
        "tracking": got["mean_tracking_error_mps"] is not None and got["mean_tracking_error_mps"] <= c["max_mean_tracking_error_mps"],
        "falls": got["falls"] is not None and got["falls"] <= c["max_falls"],
        "sustained_slip": got["sustained_slip_p95_mps"] is not None and got["sustained_slip_p95_mps"] <= c["max_sustained_slip_p95_mps"],
    }
    return all(checks.values()), {"criteria": dict(c), "measured": got, "passed": checks}


# The MVP's fixed command set for the Go2 flat task: 64 commands within the training ranges, sampled once with seed
# 20261012 (python -m sim2gate.checks.commands --n 64 --seed 20261012).
GO2_COMMANDS = os.path.join(os.path.dirname(__file__), "go2_flat_commands_64.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _file_record(path):
    return {"path": os.path.abspath(path), "sha256": sha256(path), "bytes": os.path.getsize(path)}


def run_configs(run_dir):
    """The env and agent configs Isaac Lab saved with the run (params/*.yaml), with their hashes."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "params", "*"))):
        if os.path.isfile(p):
            out[os.path.basename(p)] = _file_record(p)
    return out


def _callback(a):
    return ["--external_callback", "sim2gate.mvp.tasks.register"] if a.task in VARIANTS else []


def train_cmd(a, seed):
    physics = [f"physics={a.physics}"] if a.physics else []
    iters = ["--max_iterations", str(a.iterations)] if a.iterations else []
    if a.isaaclab_script:
        return isaac_python() + [a.isaaclab_script, "--headless", "--task", a.task, "--seed", str(seed)] + iters \
            + physics + list(a.train_arg)
    return isaac_python() + [train_script(), "--rl_library", "rsl_rl", "--task", a.task, "--seed", str(seed)] \
        + _callback(a) + iters + physics + list(a.train_arg)


def check2_cmd(a, ckpt, report):
    physics = [f"physics={a.physics}"] if a.physics else []
    return isaac_python() + ["-m", "sim2gate.checks.isaaclab_exploits", "--steps", str(a.steps),
                             "--commands", a.commands, "--out", report, "--",
                             "--task", a.task, "--num_envs", str(a.num_envs), "--checkpoint", ckpt] \
        + _callback(a) + physics + list(a.play_arg)


def _save(manifest, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, path)


_SHUTDOWN_FRAME = re.compile(r'simulation_app\.py", line \d+ in close')


def shutdown_crash(rc, text, ck, iterations):
    """Evidence that a training run finished and only Isaac Sim's shutdown crashed, or None.

    Seen on P0 v1 seed 3 (Oct 14): all 300 iterations ran and model_299.pt was saved, then the process died inside
    SimulationApp.close() and exited 1. All of these must hold: a non-zero exit, a known iteration count, the log's
    last training line is the final iteration, the newest checkpoint is the final one, and the crash dump shows
    SimulationApp.close() in progress. Anything else stays a training failure."""
    if rc == 0 or not iterations or not ck or not text:
        return None
    last = re.findall(r"Learning iteration +(\d+)/(\d+)", text)
    m = re.search(r"model_(\d+)\.pt$", ck)
    final = iterations - 1
    if not last or tuple(map(int, last[-1])) != (final, iterations) or not m or int(m.group(1)) != final:
        return None
    if not _SHUTDOWN_FRAME.search(text[text.rfind("Learning iteration"):]):
        return None
    return {"returncode": rc, "last_iteration": f"{final}/{iterations}", "final_checkpoint": os.path.basename(ck),
            "crash": "process died in Isaac Sim's SimulationApp.close() after training finished"}


def _still_valid(entry):
    ck = (entry or {}).get("checkpoint") or {}
    return bool(ck.get("path")) and os.path.exists(ck["path"]) and sha256(ck["path"]) == ck.get("sha256")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.mvp.baselines", description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True, help="policy set ID from the plan's test set, e.g. P0")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--commands", default=GO2_COMMANDS,
                   help="the fixed command set (default: the MVP's Go2 set shipped with Sim2Gate)")
    p.add_argument("--out", required=True)
    p.add_argument("--task", default=P0_TASK, help=f"default {P0_TASK} (plan change C4: the stock Go2 flat task plus "
                                                   f"a foot-slip penalty)")
    p.add_argument("--physics", default="isaacsim_physx",
                   help="Isaac Lab physics preset for training and check 2 (default isaacsim_physx: Isaac Sim's PhysX, "
                        "the plan's training engine; '' for the task's own default). Not 'physx': in Isaac Lab 3.0 that "
                        "is an automatic choice that picks the kitless OvPhysX engine when Isaac Sim is not launched")
    p.add_argument("--iterations", type=int, default=None, help="training iterations (default: the task's own)")
    p.add_argument("--num-envs", type=int, default=64, help="check 2 environments (must match the command set)")
    p.add_argument("--steps", type=int, default=1000, help="check 2 control steps (1000 = one 20 s episode)")
    p.add_argument("--train-arg", action="append", default=[], help="extra argument for training (repeatable)")
    p.add_argument("--play-arg", action="append", default=[], help="extra argument for check 2's play run")
    p.add_argument("--isaaclab-script", default=None, help="Isaac Lab 3.0 beta only: path to rsl_rl/train.py")
    p.add_argument("--calibrate", action="store_true", help="set thresholds from these policies (P0 seeds 1-3 only)")
    a = p.parse_args(argv)
    from ..checks.commands import load_command_set
    cmd_doc = load_command_set(a.commands)
    if len(cmd_doc["commands"]) != a.num_envs:
        p.error(f"the command set has {len(cmd_doc['commands'])} commands; --num-envs is {a.num_envs}")
    if a.calibrate and (a.policy != "P0" or sorted(a.seeds) != [1, 2, 3]):
        p.error("the plan sets thresholds from P0 seeds 1 to 3 only")
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    manifest.update({"policy": a.policy, "task": a.task, "task_variant": VARIANTS.get(a.task),
                     "baseline_criteria": dict(BASELINE_CRITERIA), "physics": a.physics or None,
                     "train_args": list(a.train_arg), "play_args": list(a.play_arg),
                     "command_set": {"path": os.path.abspath(a.commands), "sha256": cmd_doc["sha256"],
                                     "seed": cmd_doc.get("seed"), "n": len(cmd_doc["commands"])},
                     "check2": {"steps": a.steps, "num_envs": a.num_envs}})
    manifest.setdefault("seeds", {})
    try:
        import importlib.metadata as md
        manifest["versions"] = {k: md.version(k) for k in ("sim2gate", "rsl-rl-lib")}
    except Exception:
        pass
    ok = True
    for seed in a.seeds:
        key = str(seed)
        entry = manifest["seeds"].get(key) or {}
        tag = f"{a.policy}_s{seed}"
        if _still_valid(entry):
            print(f"[sim2gate] {tag}: checkpoint already trained and unchanged, not retraining")
        else:
            entry = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
            before = _run_dirs()
            cmd = train_cmd(a, seed)
            rc, text = run(cmd, os.path.join(a.out, f"train_{tag}.log"))
            new = sorted(_run_dirs() - before)
            entry.update({"train_cmd": cmd[1:], "train_returncode": rc,
                          "run_dir": os.path.abspath(new[0]) if len(new) == 1 else None})
            ck = _last_checkpoint(new[0]) if len(new) == 1 else None
            crash = shutdown_crash(rc, text, ck, a.iterations or (VARIANTS.get(a.task) or {}).get("max_iterations"))
            if crash:
                entry["train_shutdown_crash"] = crash
                print(f"[sim2gate] {tag}: training finished ({crash['last_iteration']}, {crash['final_checkpoint']}); "
                      f"Isaac Sim crashed while shutting down (exit {rc}). Using the final checkpoint.")
            elif rc != 0 or ck is None:
                entry["error"] = (f"training exited {rc}" if rc else "") or \
                    f"expected one new run folder with a checkpoint, found {len(new)} folder(s)"
                manifest["seeds"][key] = entry
                _save(manifest, mpath)
                ok = False
                continue
            entry["checkpoint"] = _file_record(ck)
            entry["configs"] = run_configs(new[0])
            manifest["seeds"][key] = entry
            _save(manifest, mpath)
        report = os.path.join(a.out, f"check2_{tag}.json")
        cmd = check2_cmd(a, entry["checkpoint"]["path"], report)
        if os.path.exists(report):
            os.remove(report)
        rc, _ = run(cmd, os.path.join(a.out, f"check2_{tag}.log"))
        entry["check2_returncode"] = rc
        if os.path.exists(report):
            rep = json.load(open(report))
            got = (rep.get("checkpoint") or {}).get("sha256")
            entry["check2_report"] = _file_record(report)
            entry["acceptance"] = dict(zip(("accepted", "details"), baseline_acceptance(rep)))
            entry.pop("error", None)
            if got != entry["checkpoint"]["sha256"]:
                entry["error"] = f"check 2 measured checkpoint {got}, not {entry['checkpoint']['sha256']}"
                ok = False
        else:
            entry["error"] = f"check 2 wrote no report (exit {rc})"
            ok = False
        manifest["seeds"][key] = entry
        _save(manifest, mpath)
    rejected = [s for s in a.seeds if not ((manifest["seeds"].get(str(s)) or {}).get("acceptance") or {}).get("accepted")]
    if a.calibrate and ok and rejected:
        print(f"[sim2gate] not calibrating: seeds {rejected} do not meet the baseline criteria "
              f"(manifest: seeds.<seed>.acceptance)", file=sys.stderr)
        manifest["calibration"] = {"error": f"seeds {rejected} do not meet the baseline criteria"}
        _save(manifest, mpath)
        ok = False
    if a.calibrate and ok:
        from ..checks.calibrate import CalibrationError, calibrate
        th = os.path.join(a.out, "thresholds.json")
        reports = [manifest["seeds"][str(s)]["check2_report"]["path"] for s in a.seeds]
        try:
            doc = calibrate([(r, open(r, "rb").read()) for r in reports])
        except CalibrationError as e:
            print(f"[sim2gate] calibration refused: {e}", file=sys.stderr)
            manifest["calibration"] = {"error": str(e)}
            ok = False
        else:
            if os.path.exists(th):
                print(f"[sim2gate] {th} exists; not overwriting a thresholds file", file=sys.stderr)
                ok = False
            else:
                with open(th, "w") as f:
                    json.dump(doc, f, indent=1)
                manifest["calibration"] = {"thresholds": _file_record(th), "thresholds_values": doc["thresholds"],
                                           "baseline_maxima": doc["baseline_maxima"]}
        _save(manifest, mpath)
    print("\n" + "=" * 80)
    for s in a.seeds:
        e = manifest["seeds"].get(str(s), {})
        state = "ERROR " + e["error"] if e.get("error") else \
            ("ok (Isaac Sim crashed at shutdown after training finished)" if e.get("train_shutdown_crash") else "ok")
        ck = (e.get("checkpoint") or {}).get("sha256", "")[:12]
        acc = e.get("acceptance") or {}
        m = (acc.get("details") or {}).get("measured") or {}
        fmt = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        verdict = {True: "ACCEPTED", False: "NOT ACCEPTED"}.get(acc.get("accepted"), "not assessed")
        print(f"{a.policy} seed {s}: {state}  checkpoint {ck}  baseline {verdict}: tracking error "
              f"{fmt(m.get('mean_tracking_error_mps'))} m/s, falls {fmt(m.get('falls'))}, sustained slip p95 "
              f"{fmt(m.get('sustained_slip_p95_mps'))} m/s")
    if a.calibrate:
        cal = manifest.get("calibration") or {}
        for m, v in (cal.get("thresholds_values") or {}).items():
            print(f"  threshold {m:40s} {v:.6g}")
    print(f"\n{'BASELINES DONE' if ok else 'BASELINES INCOMPLETE'}  (manifest: {mpath})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
