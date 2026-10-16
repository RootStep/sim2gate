"""The fixed velocity command set (MVP plan: "one fixed command set used everywhere, sampled once with a recorded
seed (forward, lateral and yaw rates within the task's training range)").

    python -m sim2gate.checks.commands --n 64 --seed 20261012 --out go2_flat_commands_64.json

A command set is a JSON file: {"version": 1, "seed", "ranges": {lin_vel_x, lin_vel_y, ang_vel_z: [lo, hi]},
"commands": [[vx, vy, wz], ...] (one per environment), "sha256"} where sha256 covers the canonical JSON of the
commands, so two runs can be shown to have used the same set.

`pin(env, commands)` makes an Isaac Lab velocity command term (UniformVelocityCommand) hold environment i at
commands[i] for the whole run: every resample (at reset and every resampling interval) writes the fixed value, and
heading control and standing environments are switched off for every environment, so the yaw rate is the fixed
value too.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import torch

# Isaac Lab 3.0 Go2 velocity task training ranges (velocity_env_cfg.CommandsCfg); the task's own ranges are used
# when the command term is available.
DEFAULT_RANGES = {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)}
AXES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


def _sha(commands):
    canon = json.dumps([[float(x) for x in row] for row in commands], separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def make_command_set(n, seed, ranges=None):
    """n commands, uniform within the ranges, from a CPU generator seeded with `seed` (reproducible anywhere)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    ranges = {k: tuple(float(v) for v in (ranges or DEFAULT_RANGES)[k]) for k in AXES}
    for k, (lo, hi) in ranges.items():
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError(f"bad range for {k}: {(lo, hi)}")
    g = torch.Generator().manual_seed(int(seed))
    cols = [lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64) for lo, hi in (ranges[k] for k in AXES)]
    commands = [[round(float(c[i]), 6) for c in cols] for i in range(n)]
    return {"version": 1, "seed": int(seed), "ranges": {k: list(v) for k, v in ranges.items()},
            "commands": commands, "sha256": _sha(commands)}


def load_command_set(path):
    doc = json.load(open(path))
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("commands"), list):
        raise ValueError(f"{path}: not a command set file")
    cmds = doc["commands"]
    if not cmds or any(not isinstance(r, list) or len(r) != 3 or
                       not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in r)
                       for r in cmds):
        raise ValueError(f"{path}: commands must be a non-empty list of [vx, vy, wz]")
    if doc.get("sha256") != _sha(cmds):
        raise ValueError(f"{path}: sha256 does not match the commands (edited?)")
    return doc


def summary(doc, source):
    return {"source": source, "seed": doc.get("seed"), "n": len(doc["commands"]), "ranges": doc.get("ranges"),
            "sha256": doc["sha256"]}


def term_ranges(term):
    r = getattr(getattr(term, "cfg", None), "ranges", None)
    if r is None:
        return None
    try:
        return {k: tuple(float(v) for v in getattr(r, k)) for k in AXES}
    except (AttributeError, TypeError):
        return None


def pin(env, doc, term_name="base_velocity", tile=False):
    """Hold each environment's velocity command at the command set's value. Returns (undo, info) or raises.

    tile: the environment count may be a whole multiple of the command count; environment i then gets command
    i mod n (used by check 1, where each sweep group of n episodes runs the same n commands)."""
    cm = getattr(env, "command_manager", None)
    if cm is None:
        raise RuntimeError("the environment has no command manager")
    term = cm.get_term(term_name)
    for attr in ("vel_command_b", "is_heading_env", "is_standing_env", "_resample_command"):
        if not hasattr(term, attr):
            raise RuntimeError(f"command term {term_name!r} has no {attr}; not a velocity command term")
    n = term.vel_command_b.shape[0]
    m = len(doc["commands"])
    if (m != n) and not (tile and m and n % m == 0):
        raise ValueError(f"the command set has {m} commands, the environment has {n} envs"
                         + (" (not a whole multiple)" if tile else ""))
    fixed = torch.tensor(doc["commands"], dtype=term.vel_command_b.dtype, device=term.vel_command_b.device)
    if n != m:
        fixed = fixed.repeat(n // m, 1)
    rng = term_ranges(term)
    outside = []
    if rng is not None:
        for j, k in enumerate(AXES):
            lo, hi = rng[k]
            if bool(((fixed[:, j] < lo - 1e-9) | (fixed[:, j] > hi + 1e-9)).any()):
                outside.append(k)
    orig = term._resample_command

    def _resample_command(env_ids):
        orig(env_ids)                                    # keeps the term's own bookkeeping (and its RNG use)
        term.vel_command_b[env_ids] = fixed[env_ids]
        term.is_heading_env[env_ids] = False
        term.is_standing_env[env_ids] = False

    term._resample_command = _resample_command
    _resample_command(torch.arange(n, device=term.vel_command_b.device))
    info = {"term": term_name, "task_ranges": {a: list(v) for a, v in rng.items()} if rng else None,
            "outside_task_ranges": outside, "tiled": n // m}

    def undo():
        if term.__dict__.get("_resample_command") is _resample_command:
            del term._resample_command
    return undo, info


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.checks.commands", description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, required=True, help="number of commands (one per environment)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    for k in AXES:
        p.add_argument(f"--{k.replace('_', '-')}", type=float, nargs=2, default=DEFAULT_RANGES[k], metavar=("LO", "HI"))
    a = p.parse_args(argv)
    doc = make_command_set(a.n, a.seed, {k: getattr(a, k) for k in AXES})
    if os.path.exists(a.out):
        print(f"[sim2gate] {a.out} exists; not overwriting a command set", file=sys.stderr)
        return 2
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"[sim2gate] {a.n} commands, seed {a.seed}, sha256 {doc['sha256'][:12]} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
