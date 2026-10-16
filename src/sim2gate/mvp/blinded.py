"""The blinded run (MVP plan, steps 4 and 5): checks 2, 1 and 5 on every test-set policy, seen only by ID.

Vihaan, holding the policies (private list, not in the repository):

    python -m sim2gate.mvp.blinded prepare --policies private.json --blind blind --key ~/sim2gate_key.json

  private.json: {"policies": [{"policy": "P3", "seed": 2, "checkpoint": "logs/.../model_1999.pt",
                               "task": "Sim2Gate-Velocity-Flat-Go2-P3-v1"}, ...]}
  It gives each policy a random ID, copies its checkpoint to blind/<ID>/model.pt, and writes blind/manifest.json
  (IDs and checkpoint hashes only), blind/aliases.json (each ID's task alias -> an opaque task token) and the key
  (ID -> policy and seed), which must be kept outside the blind folder.

Whoever runs the checks (inside the Isaac Lab container):

    python -m sim2gate.mvp.blinded run --blind blind --out mvp_runs/blinded
    python -m sim2gate.mvp.blinded seal --out mvp_runs/blinded       # all verdicts recorded: prints their digest
    python -m sim2gate.mvp.blinded replay --blind blind --out mvp_runs/blinded       # A6, after sealing

  Each policy runs in its own task (the simulator it was trained in), named only by its alias Sim2Gate-Blind-<ID>,
  under the frozen thresholds and success reference shipped with this package. `run` resumes: a measurement whose
  output exists is kept. `seal` writes verdicts.sha256 and prints one digest; send it to Vihaan, who releases the
  key only then. Scoring (sim2gate.mvp.acceptance) is the only step that reads the key.
"""
import argparse
import datetime
import json
import os
import secrets
import shutil
import sys

from ..checks import robustness, transfer
from ..training.smoke_go2 import isaac_python, run
from . import acceptance, report, tasks, verdict
from .baselines import GO2_COMMANDS, sha256

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_THRESHOLDS = os.path.join(HERE, "frozen_thresholds_p0v4.json")
FROZEN_SUCCESS = os.path.join(HERE, "frozen_success_p0v4.json")
REPLAY_SEED = 20261015                   # A6: which 3 policies are rerun, fixed before the run
CALLBACK = ["--external_callback", "sim2gate.mvp.tasks.register"]
CHECK2 = {"steps": 1000, "num_envs": 64, "physics": transfer.PHYSX}
FIXED = list(transfer.FIXED_OVERRIDES) + list(transfer.ORDERING)
MEASUREMENTS = ("check2", "sweep", "physx", "newton", "joint")
REPORT_ONLY = ("joint",)                 # measured and reported, never part of the verdict


def alias(pid):
    return f"Sim2Gate-Blind-{pid}"


def _save(doc, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def _inside(path, folder):
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return path == folder or path.startswith(folder + os.sep)


# ---- prepare (Vihaan) ------------------------------------------------------------------------------------------

def prepare(policies, blind, key_path, rng=None):
    rng = rng or secrets.SystemRandom()
    if _inside(key_path, blind):
        raise ValueError("the key must be kept outside the blind folder")
    if os.path.exists(os.path.join(blind, "manifest.json")) or os.path.exists(key_path):
        raise ValueError("a blind folder or key already exists here; not overwriting")
    seen = set()
    for p in policies:
        if p["task"] not in tasks.VARIANTS:
            raise ValueError(f"unknown task {p['task']!r}")
        if (p["policy"], p["seed"]) in seen:
            raise ValueError(f"{p['policy']} seed {p['seed']} listed twice")
        seen.add((p["policy"], p["seed"]))
        if not os.path.isfile(p["checkpoint"]):
            raise ValueError(f"checkpoint not found: {p['checkpoint']}")
    ids = set()
    while len(ids) < len(policies):
        ids.add("B" + "".join(rng.choice("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(4)))
    ids = sorted(ids)
    rng.shuffle(ids)
    manifest = {"version": 1, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "policies": {}}
    aliases, key = {}, {"version": 1, "policies": {}}
    for pid, p in zip(ids, policies):
        dst = os.path.join(blind, pid, "model.pt")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(p["checkpoint"], dst)
        h = sha256(dst)
        if h != sha256(p["checkpoint"]):
            raise ValueError(f"copy of {p['checkpoint']} does not match")
        manifest["policies"][pid] = {"checkpoint": f"{pid}/model.pt", "sha256": h}
        aliases[alias(pid)] = tasks.task_token(p["task"])
        key["policies"][pid] = {"policy": p["policy"], "seed": p["seed"], "task": p["task"],
                                "source_checkpoint": os.path.abspath(p["checkpoint"]), "sha256": h}
    manifest["policies"] = dict(sorted(manifest["policies"].items()))
    _save(manifest, os.path.join(blind, "manifest.json"))
    _save(aliases, os.path.join(blind, "aliases.json"))
    key["manifest_sha256"] = sha256(os.path.join(blind, "manifest.json"))
    key["policies"] = dict(sorted(key["policies"].items()))
    _save(key, key_path)
    return manifest


# ---- run ---------------------------------------------------------------------------------------------------------

def commands(pid, ckpt, out_dir, commands_path=GO2_COMMANDS):
    task = ["--task", alias(pid)]
    o = {m: os.path.join(out_dir, f"{m}.json") for m in MEASUREMENTS}
    cond = lambda mode, physics, out: isaac_python() + [  # noqa: E731
        "-m", "sim2gate.checks.isaaclab_conditions", "--mode", mode, "--commands", commands_path, "--out", out,
        "--"] + task + ["--checkpoint", ckpt] + CALLBACK + [physics] + FIXED
    return {
        "check2": isaac_python() + ["-m", "sim2gate.checks.isaaclab_exploits", "--steps", str(CHECK2["steps"]),
                                    "--commands", commands_path, "--out", o["check2"], "--"] + task
        + ["--num_envs", str(CHECK2["num_envs"]), "--checkpoint", ckpt] + CALLBACK + [CHECK2["physics"]],
        "sweep": cond("sweep", transfer.PHYSX, o["sweep"]),
        "physx": cond("nominal", transfer.PHYSX, o["physx"]),
        "newton": cond("nominal", transfer.NEWTON, o["newton"]),
        "joint": cond("joint", transfer.PHYSX, o["joint"]),
    }, o


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def measure(pid, entry, blind, out, runner=None):
    """Run the four measurements of one policy (keeping any already done) and write its verdict."""
    runner = runner or run
    ckpt = os.path.abspath(os.path.join(blind, entry["checkpoint"]))
    d = os.path.join(out, "policies", pid)
    os.makedirs(d, exist_ok=True)
    cmds, outs = commands(pid, ckpt, d)
    problems, report_problems = [], []
    if sha256(ckpt) != entry["sha256"]:
        problems.append("the checkpoint does not match the blind manifest")
    else:
        for m in MEASUREMENTS:
            if os.path.exists(outs[m]):
                continue
            rc, _ = runner(cmds[m], os.path.join(d, f"{m}.log"))
            if not os.path.exists(outs[m]):
                err = outs[m] + ".error.txt"
                tail = [x for x in open(err).read().splitlines() if x.strip()][-1:] if os.path.exists(err) else []
                msg = f"{m}: no output (exit {rc}){': ' + tail[0] if tail else ''}"
                (report_problems if m in REPORT_ONLY else problems).append(msg)
    data = {k: _load(outs[m]) for k, m in (("check2_report", "check2"), ("sweep_raw", "sweep"),
                                           ("physx_raw", "physx"), ("newton_raw", "newton"), ("joint_raw", "joint"))}
    success = json.load(open(FROZEN_SUCCESS))
    v = verdict.build(pid, thresholds=json.load(open(FROZEN_THRESHOLDS)), success=success, **data)
    if data["joint_raw"] is not None:                 # the full report-only analysis, with Arena's dataset table
        _save(robustness.joint_report(data["joint_raw"], success), os.path.join(d, "joint_report.json"))
    measured = {v["checkpoint_sha256"]} - {None}
    if measured and measured != {entry["sha256"]}:
        problems.append("the measurements are of a different checkpoint than the blind manifest names")
    if problems:
        v["problems"] = problems + v["problems"]
        v["verdict"], v["exit_code"] = "incomplete", 2
    if report_problems:                               # a report-only measurement never blocks the verdict
        v["report_only"]["problems"] = report_problems
    v["frozen"] = {"thresholds_sha256": sha256(FROZEN_THRESHOLDS), "success_sha256": sha256(FROZEN_SUCCESS)}
    v["measurements"] = {m: {"path": os.path.relpath(outs[m], out), "sha256": sha256(outs[m])}
                         for m in MEASUREMENTS if os.path.exists(outs[m])}
    return v


def _env(blind):
    os.environ[tasks.BLIND_ALIASES_ENV] = os.path.abspath(os.path.join(blind, "aliases.json"))


def run_all(blind, out, runner=None, only=None):
    _env(blind)
    man = json.load(open(os.path.join(blind, "manifest.json")))
    vdir = os.path.join(out, "verdicts")
    if os.path.exists(os.path.join(out, "verdicts.sha256")):
        raise ValueError("this run is sealed; verdicts can no longer change")
    done = {}
    for pid, entry in man["policies"].items():
        if only and pid not in only:
            continue
        v = measure(pid, entry, blind, out, runner)
        _save(v, os.path.join(vdir, f"{pid}.json"))
        report.write(v, os.path.join(out, "reports", f"{pid}.html"))
        done[pid] = v
        print(f"[sim2gate] {pid}: {v['verdict']} (exit {v['exit_code']})", flush=True)
    return done


def seal(out, blind):
    man = json.load(open(os.path.join(blind, "manifest.json")))
    vdir = os.path.join(out, "verdicts")
    missing = [p for p in man["policies"] if not os.path.exists(os.path.join(vdir, f"{p}.json"))]
    if missing:
        raise ValueError(f"no verdict yet for {', '.join(missing)}")
    incomplete = [p for p in man["policies"] if json.load(open(os.path.join(vdir, f"{p}.json")))["exit_code"] == 2]
    if incomplete:
        raise ValueError(f"incomplete verdicts (rerun them first): {', '.join(incomplete)}")
    path = os.path.join(out, "verdicts.sha256")
    if os.path.exists(path):
        raise ValueError("already sealed")
    lines = [f"{sha256(os.path.join(vdir, f'{p}.json'))}  verdicts/{p}.json" for p in sorted(man["policies"])]
    lines.append(f"{sha256(os.path.join(blind, 'manifest.json'))}  blind/manifest.json")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return sha256(path)


def replay(blind, out, runner=None):
    if not os.path.exists(os.path.join(out, "verdicts.sha256")):
        raise ValueError("seal the verdicts before the replays (A6 compares against the sealed verdicts)")
    man = json.load(open(os.path.join(blind, "manifest.json")))
    pick = acceptance.pick_replays(man["policies"], REPLAY_SEED)
    _env(blind)
    rout = os.path.join(out, "replay")
    for pid in pick:
        v = measure(pid, man["policies"][pid], blind, rout, runner)
        _save(v, os.path.join(out, "replay_verdicts", f"{pid}.json"))
        print(f"[sim2gate] replay {pid}: {v['verdict']} (exit {v['exit_code']})", flush=True)
    return pick


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.mvp.blinded", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--policies", required=True)
    pp.add_argument("--blind", required=True)
    pp.add_argument("--key", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--blind", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--only", nargs="+", default=None, help="these IDs only")
    ps = sub.add_parser("seal")
    ps.add_argument("--blind", required=True)
    ps.add_argument("--out", required=True)
    pl = sub.add_parser("replay")
    pl.add_argument("--blind", required=True)
    pl.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "prepare":
            m = prepare(json.load(open(a.policies))["policies"], a.blind, a.key)
            print(f"[sim2gate] {len(m['policies'])} policies blinded into {a.blind}; key written to {a.key} "
                  f"(keep it outside the blind folder and away from whoever runs the checks)")
        elif a.cmd == "run":
            vs = run_all(a.blind, a.out, only=a.only)
            bad = [pid for pid, v in vs.items() if v["exit_code"] == 2]
            print(f"\nBLINDED RUN {'DONE' if not bad else 'INCOMPLETE: ' + ', '.join(bad)}  ({len(vs)} policies)")
            return 0 if not bad else 1
        elif a.cmd == "seal":
            d = seal(a.out, a.blind)
            print(f"VERDICTS SEALED. Digest to send to Vihaan before he releases the key:\n{d}")
        else:
            pick = replay(a.blind, a.out)
            print(f"\nREPLAYS DONE: {', '.join(pick)}")
    except ValueError as e:
        print(f"[sim2gate] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
