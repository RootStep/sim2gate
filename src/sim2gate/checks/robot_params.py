"""Read back the robot as the physics engine actually holds it, and compare two engines (check 5 parity: the same
robot on both).

    # one dump per engine (inside the Isaac Lab container; any trained checkpoint of the task)
    python -m sim2gate.checks.robot_params --out robot_physx.json -- --task <task> --num_envs 4 \\
        --checkpoint <model.pt> physics=isaacsim_physx env.sim.use_newton_actuators=false
    python -m sim2gate.checks.robot_params --out robot_newton.json -- ... physics=newton_mjwarp ...
    # the differences
    python -m sim2gate.checks.robot_params --compare robot_physx.json robot_newton.json

The environment is built under check 1/5's nominal profile (no startup mass or centre-of-mass randomization,
whose random draws would differ between engines). The dump is taken on the first control step, from what the engine reports (Isaac Lab's articulation data and actuator
models), for environment 0: joint and body order, default pose, joint stiffness / damping / armature / friction /
limits, body masses, inertias and centres of mass, the actuator models' gains and limits, and the simulation step.
Then the robot stands for `--stand` control steps with zero actions (the default joint targets) and the base height
and joint tracking are recorded: a static robot that settles differently on two engines is not the same robot,
whatever the parameters say.
"""
import argparse
import json
import math
import os
import runpy
import sys

import torch

DATA_FIELDS = ("default_joint_pos", "joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff",
               "joint_pos_limits", "soft_joint_pos_limits", "joint_vel_limits", "joint_effort_limits",
               "body_mass", "body_inertia", "body_com_pos_b")
ACTUATOR_FIELDS = ("stiffness", "damping", "actuator_effort_limit", "actuator_velocity_limit", "_saturation_effort",
                   "armature", "friction")


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def _env0(x):
    x = _t(x)
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        v = x[0] if x.dim() > 0 and x.shape[0] > 0 else x
        return [round(float(a), 7) for a in v.detach().flatten().cpu().tolist()]
    if isinstance(x, (int, float)):
        return [round(float(x), 7)]
    return None


def snapshot(env):
    robot = env.scene["robot"]
    d = robot.data
    out = {"joint_names": list(getattr(robot, "joint_names", []) or []),
           "body_names": list(getattr(robot, "body_names", []) or []),
           "data": {}, "actuators": {}, "missing": []}
    for f in DATA_FIELDS:
        try:
            v = _env0(getattr(d, f, None))
        except Exception as e:                     # a field one engine cannot provide is itself a difference
            v, out["missing"] = None, out["missing"] + [f"{f}: {type(e).__name__}"]
        if v is None:
            out["missing"].append(f)
        out["data"][f] = v
    for name, a in (getattr(robot, "actuators", None) or {}).items():
        rec = {"class": type(a).__name__, "joints": list(getattr(a, "joint_names", []) or [])}
        for f in ACTUATOR_FIELDS:
            try:
                rec[f] = _env0(getattr(a, f, None))
            except Exception:
                rec[f] = None
        out["actuators"][name] = rec
    cfg = env.cfg
    out["sim"] = {"dt": float(cfg.sim.dt), "decimation": int(cfg.decimation),
                  "gravity": list(getattr(cfg.sim, "gravity", []) or []),
                  "physics_manager": getattr(getattr(env.sim, "physics_manager", None), "__name__", None)}
    return out


class Reader:
    def __init__(self, out_path, stand_steps=100):
        self.out_path, self.stand_steps = out_path, stand_steps
        self.doc = None

    def step(self, wrapper, orig_step, actions):
        try:
            return self._step(wrapper, orig_step, actions)
        except KeyboardInterrupt:
            raise
        except BaseException:                 # Isaac Sim closes the process before Python prints this
            import traceback
            text = "[sim2gate] ERROR while reading the robot:\n" + traceback.format_exc()
            sys.stderr.write(text); sys.stderr.flush()
            with open(self.out_path + ".error.txt", "w") as f:
                f.write(text)
            raise

    def _step(self, wrapper, orig_step, actions):
        env = wrapper.unwrapped
        doc = snapshot(env)
        wrapper.reset()
        robot = env.scene["robot"]
        zero = torch.zeros_like(actions)
        z, err = [], []
        for _ in range(self.stand_steps):
            orig_step(wrapper, zero)
            pos = _t(robot.data.root_pos_w) if hasattr(robot.data, "root_pos_w") else _t(robot.data.root_link_pos_w)
            jp, jd = _t(robot.data.joint_pos), _t(robot.data.default_joint_pos)
            z.append(float(pos[:, 2].mean()))
            err.append(float((jp - jd).abs().mean()))
        doc["stand"] = {"steps": self.stand_steps, "base_height_m": [round(v, 5) for v in z[::10]] + [round(z[-1], 5)],
                        "final_base_height_m": round(z[-1], 5),
                        "final_mean_joint_error_rad": round(err[-1], 6)}
        doc["play_args"] = sys.argv[1:]
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"[sim2gate] robot as the engine holds it ({doc['sim']['physics_manager']}): {self.out_path}; "
              f"standing base height {doc['stand']['final_base_height_m']} m")
        self.doc = doc
        raise KeyboardInterrupt


def compare(a, b, rel=1e-3, abs_tol=1e-6):
    """Differences between two dumps: [(path, a value, b value)]."""
    diffs = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k in ("play_args", "physics_manager"):
                    continue
                walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        elif isinstance(x, list) and isinstance(y, list) and all(isinstance(v, (int, float)) for v in x + y):
            if len(x) != len(y) or any(not math.isclose(p, q, rel_tol=rel, abs_tol=abs_tol) for p, q in zip(x, y)):
                diffs.append((path, x, y))
        elif x != y:
            diffs.append((path, x, y))
    walk(a, b, "")
    return diffs


def _short(v, n=8):
    if isinstance(v, list) and len(v) > n:
        return str(v[:n])[:-1] + f", ... ({len(v)} values)]"
    return str(v)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--compare"]:
        a, b = json.load(open(argv[1])), json.load(open(argv[2]))
        d = compare(a, b)
        print(f"[sim2gate] {a['sim'].get('physics_manager')} vs {b['sim'].get('physics_manager')}: "
              f"{len(d)} difference(s)")
        for path, x, y in d:
            print(f"  {path}\n    {_short(x)}\n    {_short(y)}")
        return 0 if not d else 1
    ours, rest = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    p = argparse.ArgumentParser(prog="python -m sim2gate.checks.robot_params")
    p.add_argument("--out", required=True)
    p.add_argument("--stand", type=int, default=100, help="control steps standing with zero actions")
    a = p.parse_args(ours)
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as W
    import gymnasium
    from .isaaclab_conditions import apply_profile
    orig_make = gymnasium.make

    def make(*args, **kw):
        # the nominal profile: no startup mass / centre-of-mass randomization, whose draws differ per engine
        apply_profile(kw["cfg"])
        return orig_make(*args, **kw)
    gymnasium.make = make
    rd = Reader(a.out, a.stand)
    orig = W.step
    W.step = lambda self, actions: rd.step(self, orig, actions)
    sys.argv = ["play_rsl_rl"] + rest
    try:
        runpy.run_module("isaaclab_rl.entrypoints.backends.play_rsl_rl", run_name="__main__")
    except KeyboardInterrupt:
        pass
    return 0 if os.path.exists(a.out) else 1


if __name__ == "__main__":
    sys.exit(main())
