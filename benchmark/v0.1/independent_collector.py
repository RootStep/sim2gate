#!/usr/bin/env python3
"""Tier-R trajectory collector: run one policy under one condition cell of the G2 manifest in the common P0 v5b
task, and write every episode's raw per-step trajectory for the independent truth code (independent_truth.py).

    /isaac-sim/python.sh benchmark/v0.1/independent_collector.py --manifest G2.json --policy s306 --cell F040 \\
        --out runs/s306__F040.jsonl.gz -- --task Sim2Gate-Velocity-Flat-Go2-P0-v5b --checkpoint <model_1999.pt> \\
        physics=isaacsim_physx env.sim.use_newton_actuators=false --external_callback sim2gate.v2.tasks.register

Environment setup reuses Sim2Gate's PLUMBING so the common task matches the historical nominal conditions exactly:
  - apply_profile: friction 0.8/0.6, startup mass/COM/payload randomization and pushes removed, task observation
    noise on, swept DC motor at zero delay;
  - command pinning: environment i holds command i mod n of the manifest's command set;
  - install: gymnasium.make / wrapper.step / runner.load hooks around Isaac Lab's play entry point.
It imports NO Sim2Gate scoring code or calibrated limits (robustness scoring, exploits_score, layer1_limits,
transfer, frozen threshold files). Truth is computed later, offline, from the raw trajectories.

Additionally, per cell:
  - static (dynamic = 0.75x) contact friction set on the task's physics_material event before the env is built;
  - every termination except the time limit is removed, so a fallen robot keeps being recorded and the
    independent fall detector, not the environment, decides falls;
  - observation or action latency of whole control steps (independent_truth.LatencyBuffer).
Each environment's first episode is recorded until the time limit. Written as JSON lines, gzip: a header record,
then one `independent-truth-episode-v1` record per environment.

STATUS: not yet run. Running it (even as a smoke test) uses the GPU and needs explicit approval with G2.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import independent_truth as it  # noqa: E402

FOOT_PATTERN = ".*_foot"
CONTACT_SENSOR = "contact_forces"
ROUND = 5


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cell(manifest, cell_id):
    cells = {c["id"]: c for c in manifest["cells"]}
    if cell_id not in cells:
        raise SystemExit(f"cell {cell_id!r} not in the manifest ({', '.join(sorted(cells))})")
    c = cells[cell_id]
    for k in ("static_friction", "latency_steps", "latency_side"):
        if k not in c:
            raise SystemExit(f"cell {cell_id}: missing {k}")
    if c["latency_side"] not in ("none", "observation", "action"):
        raise SystemExit(f"cell {cell_id}: latency_side must be none, observation or action")
    return c


class EpisodeLog:
    """Accumulates per-environment samples (plain Python lists) and emits independent-truth episode records.
    Pure Python, so its behaviour is unit-tested without Isaac Lab."""

    def __init__(self, n_envs, foot_names, dt):
        self.n, self.feet, self.dt = n_envs, list(foot_names), float(dt)
        self.samples = [[] for _ in range(n_envs)]
        self.open = [True] * n_envs

    def add(self, snap, done=None):
        """snap: dict of per-env lists for one control step; done: per-env bool, True if this step ended the
        episode (its sample may already be post-reset and is not recorded)."""
        for i in range(self.n):
            if not self.open[i]:
                continue
            if done is not None and done[i]:
                self.open[i] = False
                continue
            self.samples[i].append({
                "h": snap["base_height_m"][i], "g": snap["projected_gravity_b"][i],
                "v": snap["base_lin_vel_b_xy"][i], "c": snap["cmd_vel_xy"][i], "p": snap["base_pos_w_xy"][i],
                "f": snap["foot_force_n"][i], "q": snap["foot_pos_w_xy"][i]})

    @property
    def all_closed(self):
        return not any(self.open)

    def record(self, i):
        s = self.samples[i]
        r = lambda x: round(float(x), ROUND)  # noqa: E731
        return {"schema": it.SCHEMA_VERSION, "env": i, "dt": self.dt,
                "base_height_m": [r(x["h"]) for x in s],
                "projected_gravity_b": [[r(v) for v in x["g"]] for x in s],
                "base_lin_vel_b_xy": [[r(v) for v in x["v"]] for x in s],
                "cmd_vel_xy": [[r(v) for v in x["c"]] for x in s],
                "base_pos_w_xy": [[r(v) for v in x["p"]] for x in s],
                "feet": {name: {"contact_force_n": [r(x["f"][j]) for x in s],
                                "pos_w_xy": [[r(v) for v in x["q"][j]] for x in s]}
                         for j, name in enumerate(self.feet)}}

    def write(self, path, header):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = str(path) + ".tmp"
        with gzip.open(tmp, "wt") as f:
            f.write(json.dumps({"header": header}) + "\n")
            for i in range(self.n):
                rec = self.record(i)
                it.validate_episode(rec)
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        os.replace(tmp, path)


class Collector:
    """Hooks for sim2gate.checks.isaaclab_conditions.install: on_env_cfg, step, record_checkpoint."""

    def __init__(self, manifest, manifest_path, policy_id, cell, out_path, play_args):
        self.manifest, self.policy_id, self.cell, self.out = manifest, policy_id, cell, out_path
        self.manifest_sha256 = _sha_file(manifest_path)
        self.play_args = list(play_args)
        cs = manifest["command_set"]
        from sim2gate.checks import commands as command_sets
        self.command_sets = command_sets
        self.command_doc = command_sets.load_command_set(str(HERE / cs["path"]))
        if self.command_doc["sha256"] != cs["sha256"]:
            raise SystemExit("command set sha256 differs from the manifest")
        self.n_envs = len(self.command_doc["commands"]) * int(manifest["episodes_per_cell_repeat"])
        self.checkpoints, self.cfg_changes, self.started = [], {}, False
        self.obs_buf = self.act_buf = None

    def record_checkpoint(self, path):
        try:
            self.checkpoints.append({"path": os.path.abspath(path), "sha256": _sha_file(path)})
        except OSError as e:
            self.checkpoints.append({"path": str(path), "sha256": None, "error": str(e)})

    # -- before the env is built --
    def on_env_cfg(self, cfg):
        from sim2gate.checks.isaaclab_conditions import apply_profile
        self.cfg_changes["profile_record"] = apply_profile(cfg)
        cfg.scene.num_envs = self.n_envs
        mu = float(self.cell["static_friction"])
        pm = cfg.events.physics_material.params
        pm["static_friction_range"] = (mu, mu)
        pm["dynamic_friction_range"] = (0.75 * mu, 0.75 * mu)
        self.cfg_changes["contact_material"] = {"static": mu, "dynamic": 0.75 * mu}
        removed = []
        terms = cfg.terminations
        for name in list(vars(terms)):
            if name != "time_out" and getattr(terms, name, None) is not None and not name.startswith("_"):
                setattr(terms, name, None)
                removed.append(name)
        if getattr(terms, "time_out", None) is None:
            raise RuntimeError("the task has no time_out termination; episodes would never end")
        self.cfg_changes["removed_terminations"] = removed

    # -- every policy step --
    def step(self, wrapper, orig_step, actions):
        import torch
        env = wrapper.unwrapped
        if not self.started:
            self._setup(env)
            self.started = True
            obs, extras = wrapper.reset()
            obs = self._delay_obs(obs)
            n = wrapper.num_envs
            return obs, torch.zeros(n, device=env.device), torch.zeros(n, dtype=torch.long, device=env.device), \
                dict(extras or {}, time_outs=torch.zeros(n, dtype=torch.bool, device=env.device))
        if self.act_buf is not None:
            actions = self.act_buf(actions.clone())
        out = orig_step(wrapper, actions)
        dones = out[2].bool().tolist()
        self.log.add(self._snapshot(env), done=dones)
        self.steps += 1
        if self.log.all_closed or self.steps >= self.max_steps + 5:
            self._finish(env)
            raise KeyboardInterrupt
        return (self._delay_obs(out[0]),) + tuple(out[1:])

    def _delay_obs(self, obs):
        return self.obs_buf(obs.clone()) if self.obs_buf is not None else obs

    def _setup(self, env):
        if env.num_envs != self.n_envs:
            raise RuntimeError(f"{env.num_envs} environments, expected {self.n_envs}")
        self._undo_cmd, self.cmd_info = self.command_sets.pin(env, self.command_doc, tile=True)
        robot, sensor = env.scene["robot"], env.scene.sensors[CONTACT_SENSOR]
        self.foot_ids_robot, self.foot_names = robot.find_bodies(FOOT_PATTERN)
        self.foot_ids_sensor, sensor_names = sensor.find_bodies(FOOT_PATTERN)
        if list(sensor_names) != list(self.foot_names) or len(self.foot_names) != 4:
            raise RuntimeError(f"foot bodies differ: robot {self.foot_names}, sensor {sensor_names}")
        k, side = int(self.cell["latency_steps"]), self.cell["latency_side"]
        if side == "observation" and k > 0:
            self.obs_buf = it.LatencyBuffer(k)
        elif side == "action" and k > 0:
            self.act_buf = it.LatencyBuffer(k)
        self.log = EpisodeLog(env.num_envs, self.foot_names, float(env.step_dt))
        self.steps, self.max_steps = 0, int(env.max_episode_length)

    def _snapshot(self, env):
        t = lambda x: x.torch if hasattr(x, "torch") else x  # noqa: E731  (warp arrays in Isaac Lab 3.x)
        d = env.scene["robot"].data
        pos = t(d.root_pos_w)
        feet_pos = t(d.body_pos_w)[:, self.foot_ids_robot, :2]
        forces = t(env.scene.sensors[CONTACT_SENSOR].data.net_forces_w)[:, self.foot_ids_sensor, 2]
        cmd = env.command_manager.get_term("base_velocity").vel_command_b[:, :2]
        return {"base_height_m": pos[:, 2].tolist(), "base_pos_w_xy": pos[:, :2].tolist(),
                "projected_gravity_b": t(d.projected_gravity_b).tolist(),
                "base_lin_vel_b_xy": t(d.root_lin_vel_b)[:, :2].tolist(), "cmd_vel_xy": cmd.tolist(),
                "foot_force_n": forces.tolist(), "foot_pos_w_xy": feet_pos.tolist()}

    def _finish(self, env):
        ck = self.checkpoints[-1] if self.checkpoints else None
        expected = self.manifest["policies"][self.policy_id]["checkpoint_sha256"]
        header = {"tool": "independent_collector", "tool_sha256": _sha_file(__file__),
                  "truth_lib_sha256": _sha_file(HERE / "independent_truth.py"),
                  "manifest_sha256": self.manifest_sha256, "policy_id": self.policy_id, "cell": self.cell,
                  "checkpoint": ck, "checkpoint_matches_manifest": bool(ck and ck.get("sha256") == expected),
                  "task": next((self.play_args[i + 1] for i, a in enumerate(self.play_args[:-1]) if a == "--task"), None),
                  "play_args": self.play_args, "cfg_changes": self.cfg_changes, "commands": self.cmd_info,
                  "command_set_sha256": self.command_doc["sha256"], "num_envs": env.num_envs,
                  "step_dt": float(env.step_dt), "episode_steps": self.max_steps, "steps_run": self.steps,
                  "foot_names": list(self.foot_names)}
        self.log.write(self.out, header)
        print(f"[independent] {self.policy_id} {self.cell['id']}: {env.num_envs} episodes, {self.steps} steps "
              f"-> {self.out}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, rest = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(ours)
    with open(a.manifest) as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "APPROVED":
        raise SystemExit("the G2 manifest is not APPROVED; refusing to run")
    if a.policy not in manifest["policies"]:
        raise SystemExit(f"policy {a.policy!r} not in the manifest")
    task = next((rest[i + 1] for i, x in enumerate(rest[:-1]) if x == "--task"), None)
    if task != manifest["common_task"]:
        raise SystemExit(f"--task must be the common task {manifest['common_task']!r}, got {task!r}")
    if any(x == "--num_envs" or x.startswith("--num_envs=") for x in rest):
        raise SystemExit("--num_envs is set from the manifest; do not pass it")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; runs are written once")
    col = Collector(manifest, a.manifest, a.policy, load_cell(manifest, a.cell), a.out, rest)
    from sim2gate.checks.isaaclab_conditions import install
    install(col)
    import runpy
    sys.argv = ["play_rsl_rl"] + rest + ["--num_envs", str(col.n_envs)]
    try:
        runpy.run_module("isaaclab_rl.entrypoints.backends.play_rsl_rl", run_name="__main__")
    except KeyboardInterrupt:
        pass
    return 0 if os.path.exists(a.out) else 1


if __name__ == "__main__":
    sys.exit(main())
