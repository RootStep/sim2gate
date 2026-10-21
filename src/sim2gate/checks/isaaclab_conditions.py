"""Run a policy under controlled physical conditions in Isaac Lab 3.x and record every episode: the measurement
behind check 1 (robustness sweep) and check 5 (PhysX vs Newton).

    # check 1: the whole sweep grid in one run (46 groups x 64 episodes = 2,944 environments)
    python -m sim2gate.checks.isaaclab_conditions --mode sweep --commands mvp --out raw_check1.json -- \\
        --task <task> --checkpoint <model.pt> physics=isaacsim_physx env.sim.use_newton_actuators=false

    # check 5: the nominal condition only (64 environments), once per engine
    python -m sim2gate.checks.isaaclab_conditions --mode nominal --commands mvp --out raw_physx.json -- \\
        --task <task> --checkpoint <model.pt> physics=isaacsim_physx env.sim.use_newton_actuators=false

Everything after `--` goes to Isaac Lab's RSL-RL play entry point; --num_envs is set from the mode.

The nominal profile (PROFILE), applied to the environment config before the environment is built:
- the startup randomizations of base mass and centre of mass are removed, so every robot has the model's mass;
- the robot's contact material is the task's training value, static 0.8 / dynamic 0.6 (ground 1.0, multiply);
- observation noise is on at the task's own level (Isaac Lab's play mode turns it off; the plan's noise sweep
  starts from 1x the task's default noise, so nominal includes it);
- external pushes stay off (as in play mode);
- the DC motors are Isaac Lab's DCMotor model plus a per-environment command delay (in physics steps) and output
  scale, both at their nominal values (no delay, full strength) unless swept.
Initial-state randomization at reset (pose, velocity, joint positions) is kept: it is part of the episode.

Per-environment conditions are applied after the environment is built and before any measured step: friction and
added base mass through Isaac Lab's own event terms (randomize_rigid_body_material / _mass with a single value per
group), motor strength and latency through the motor model, noise by scaling each noise term per environment. Then
every environment is reset, so all episodes start together, under their condition, on the same command set
(environment i gets command i mod 64). Each environment's first episode is recorded: whether it ended before the time
limit (a fall) and its mean |commanded - actual| base-frame xy velocity, read before the step's resets.
"""
import argparse
import hashlib
import json
import os
import runpy
import sys

import torch

from . import commands as command_sets
from . import robustness

PROFILE = "sim2gate-nominal-v1"
REMOVED_EVENTS = ("add_base_mass", "add_payload", "base_com", "push_robot", "base_external_force_torque")
# add_payload: P0 v5's training payload (sim2gate.v2.tasks); absent from v1 tasks, so v1 runs are unchanged
NOMINAL_STATIC_FRICTION, NOMINAL_DYNAMIC_FRICTION = 0.8, 0.6
MAX_LAG_STEPS = 20                       # delay buffer depth, physics steps (100 ms at 200 Hz)
EXCLUDED_FROM_CFG_DIGEST = ("sim", "armature")   # physics settings: they may differ between engines (check 5)

_MOTOR_CLS = None


def swept_motor_class(DCMotor=None, DelayBuffer=None):
    """Isaac Lab's DCMotor with a per-environment command delay (physics steps) and output scale. (The arguments
    replace Isaac Lab's classes in tests.)"""
    global _MOTOR_CLS
    if DCMotor is None:
        if _MOTOR_CLS is not None:
            return _MOTOR_CLS
        from isaaclab.actuators import DCMotor
        from isaaclab.utils.buffers import DelayBuffer

    class SweptDCMotor(DCMotor):
        def __init__(self, cfg, *a, **k):
            super().__init__(cfg, *a, **k)
            n, dev = self._num_envs, self._device
            self.sim2gate_strength = torch.ones(n, 1, device=dev)
            self.sim2gate_lag = torch.zeros(n, dtype=torch.long, device=dev)
            self._sim2gate_bufs = [DelayBuffer(MAX_LAG_STEPS, n, device=dev) for _ in range(3)]

        def sim2gate_set(self, strength=None, lag=None):
            if strength is not None:
                self.sim2gate_strength = strength.reshape(-1, 1).to(self._device, torch.float32)
            if lag is not None:
                self.sim2gate_lag = lag.to(self._device, torch.long)
                for b in self._sim2gate_bufs:
                    b.set_time_lag(self.sim2gate_lag)

        def reset(self, env_ids):
            super().reset(env_ids)
            for b in self._sim2gate_bufs:
                b.reset(env_ids)

        def compute(self, control_action, joint_pos, joint_vel):
            p, v, e = self._sim2gate_bufs
            control_action.joint_positions = p.compute(control_action.joint_positions)
            control_action.joint_velocities = v.compute(control_action.joint_velocities)
            control_action.joint_efforts = e.compute(control_action.joint_efforts)
            out = super().compute(control_action, joint_pos, joint_vel)
            self.applied_effort = self.applied_effort * self.sim2gate_strength
            out.joint_efforts = self.applied_effort
            return out

    if SweptDCMotor.__mro__[1].__module__.startswith("isaaclab"):
        _MOTOR_CLS = SweptDCMotor
    return SweptDCMotor


def apply_profile(cfg, motor_cls=None):
    """Turn a velocity-task env config into the nominal profile, in place. Returns a record of what changed."""
    rec = {"profile": PROFILE, "removed_events": [], "actuators": {}}
    ev = cfg.events
    for name in REMOVED_EVENTS:
        if getattr(ev, name, None) is not None:
            setattr(ev, name, None)
            rec["removed_events"].append(name)
    pm = getattr(ev, "physics_material", None)
    if pm is None:
        raise RuntimeError("the task has no physics_material event; cannot set the nominal contact material")
    pm.params["static_friction_range"] = (NOMINAL_STATIC_FRICTION, NOMINAL_STATIC_FRICTION)
    pm.params["dynamic_friction_range"] = (NOMINAL_DYNAMIC_FRICTION, NOMINAL_DYNAMIC_FRICTION)
    pm.params["restitution_range"] = (0.0, 0.0)
    rec["contact_material"] = {"static": NOMINAL_STATIC_FRICTION, "dynamic": NOMINAL_DYNAMIC_FRICTION}
    pol = cfg.observations.policy
    pol.enable_corruption = True
    rec["observation_noise"] = "on (task default)"
    motor_cls = motor_cls or swept_motor_class()
    base = motor_cls.__mro__[1]
    for name, a in cfg.scene.robot.actuators.items():
        ct = _resolve_class(getattr(a, "class_type", None))
        if not isinstance(ct, type) or not issubclass(ct, base):
            raise RuntimeError(f"actuator group {name!r} is {ct}, not {base.__name__}; "
                               f"the motor strength and latency conditions need the DC motor model")
        a.class_type = motor_cls
        rec["actuators"][name] = motor_cls.__name__
    return rec


def _resolve_class(ct):
    """An actuator config's class_type: a class, or (Isaac Lab 3.x) a lazily resolved 'module:Class' string."""
    if isinstance(ct, str):
        try:
            from isaaclab.utils.string import string_to_callable
        except ImportError:
            import importlib

            def string_to_callable(s):
                mod, attr = s.split(":", 1)
                return getattr(importlib.import_module(mod), attr)
        return string_to_callable(str(ct))
    return ct


def _digest(d):
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in EXCLUDED_FROM_CFG_DIGEST}
        if isinstance(x, (list, tuple)):
            return [strip(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(strip(d), sort_keys=True, default=str).encode()).hexdigest()


def _versions():
    import importlib.metadata as md
    out = {}
    for k in ("isaaclab", "isaaclab_physx", "isaaclab_newton", "newton", "mujoco-warp", "mujoco_warp", "warp-lang",
              "isaacsim", "rsl-rl-lib", "sim2gate"):
        try:
            out[k] = md.version(k)
        except Exception:
            pass
    return out


def _to_dict(obj):
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        try:
            return json.loads(json.dumps(obj.to_dict(), default=str))
        except Exception:
            pass
    return str(obj)


def _overrides(play_args):
    return [a for a in (play_args or []) if "=" in a and not a.startswith("-")]


class ConditionRecorder:
    """Applies the conditions on the first step, then records each environment's first episode."""

    def __init__(self, mode, out_path, command_set, command_source, play_args, episodes=robustness.EPISODES_PER_VALUE,
                 api=None, motor_cls=None, envelope="v1"):
        self.api, self.motor_cls = api, motor_cls
        self.envelope = envelope
        robustness.params(envelope)                      # validates the name
        self.mode, self.out_path, self.play_args = mode, out_path, list(play_args or [])
        self.command_set, self.command_source = command_set, command_source
        self.episodes = episodes
        if mode == "joint":
            # report only: every env its own draw of all five conditions (robustness.joint_conditions)
            self.groups = robustness.joint_conditions()
            self.env_group = [g["group"] for g in self.groups]
        else:
            self.groups = (robustness.conditions(envelope) if mode == "sweep"
                           else robustness.conditions(envelope)[:1])
            self.env_group = robustness.layout(self.groups, episodes)
        self.profile_record = None
        self.checkpoints = []
        self.env = None
        self.started = False
        self._err = None
        self.in_step = False

    @property
    def num_envs(self):
        return len(self.env_group)

    # -- config hook (before the env is built) --
    def on_env_cfg(self, cfg):
        with self._loud("applying the nominal profile"):
            self.profile_record = apply_profile(cfg, motor_cls=self.motor_cls)
            cfg.scene.num_envs = self.num_envs

    def _loud(self, what):
        """Report any error here in full before re-raising. Inside Isaac Lab's launch_simulation block, Isaac Sim
        closes the process on an exception before Python prints the traceback, so the run would die silently."""
        rec = self

        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, et, ev, tb):
                if et is None or et is KeyboardInterrupt:
                    return False
                import traceback
                text = f"[sim2gate] ERROR while {what}:\n" + "".join(traceback.format_exception(et, ev, tb))
                sys.stderr.write(text)
                sys.stderr.flush()
                sys.stdout.flush()
                try:
                    with open(rec.out_path + ".error.txt", "w") as f:
                        f.write(text)
                        f.flush()
                        os.fsync(f.fileno())
                except OSError:
                    pass
                return False
        return _Ctx()

    # -- first step: apply the conditions and reset --
    def _setup(self, wrapper):
        env = self.env = wrapper.unwrapped
        if self.profile_record is None:
            raise RuntimeError("the nominal profile was not applied (the environment was built before the hook)")
        if env.num_envs != self.num_envs:
            raise RuntimeError(f"{env.num_envs} environments, the {self.mode} layout needs {self.num_envs}")
        self._undo_cmd, self.cmd_info = command_sets.pin(env, self.command_set, tile=True)
        dev = env.device
        group = torch.tensor(self.env_group, device=dev)
        # float64: the grid's values compare exactly (0.8 in float32 is 0.800000012, not the nominal 0.8)
        cond = {k: torch.tensor([self.groups[g]["condition"][k] for g in self.env_group], device=dev,
                                dtype=torch.float64) for k in robustness.PARAMS}
        self.applied = apply_conditions(env, cond, group, self.groups, api=self.api)
        mgr = getattr(env, "reward_manager", None)
        if mgr is not None and callable(getattr(mgr, "compute", None)):
            orig = mgr.compute

            def compute(*a, **k):
                out = orig(*a, **k)
                if self.in_step:
                    self._err = tracking_error(env)
                return out
            mgr.compute = compute
        n = env.num_envs
        self.done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fell = torch.zeros(n, dtype=torch.bool, device=dev)
        self.err_sum = torch.zeros(n, dtype=torch.float64, device=dev)
        self.err_n = torch.zeros(n, dtype=torch.long, device=dev)
        self.steps = 0
        self.max_steps = int(env.max_episode_length)

    def step(self, wrapper, orig_step, actions):
        with self._loud(f"recording step {getattr(self, 'steps', 0)}"):
            return self._step(wrapper, orig_step, actions)

    def _step(self, wrapper, orig_step, actions):
        if not self.started:
            self._setup(wrapper)
            self.started = True
            obs, extras = wrapper.reset()             # every episode starts now, under its condition
            n = wrapper.num_envs
            extras = dict(extras or {})
            extras.setdefault("time_outs", torch.zeros(n, dtype=torch.bool, device=self.env.device))
            return obs, torch.zeros(n, device=self.env.device), torch.zeros(n, dtype=torch.long,
                                                                             device=self.env.device), extras
        self._err = None
        self.in_step = True
        try:
            out = orig_step(wrapper, actions)
        finally:
            self.in_step = False
        self.steps += 1
        err = self._err if self._err is not None else tracking_error(self.env)
        dones = out[2].bool()
        to = (out[3] or {}).get("time_outs") if len(out) > 3 else None
        if to is None:
            raise RuntimeError("the environment does not report time-outs; falls cannot be told from time limits")
        live = ~self.done
        self.err_sum[live] += err[live].double()
        self.err_n[live] += 1
        ended = dones & live
        self.fell |= ended & ~to.to(dones.device).bool()
        self.done |= ended
        if bool(self.done.all()) or self.steps >= self.max_steps + 5:
            self.finish()
            raise KeyboardInterrupt
        return out

    def record_checkpoint(self, path):
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            self.checkpoints.append({"path": os.path.abspath(path), "sha256": digest})
        except OSError as e:
            self.checkpoints.append({"path": str(path), "sha256": None, "error": str(e)})

    def finish(self):
        env = self.env
        mean = (self.err_sum / self.err_n.clamp(min=1)).tolist()
        episodes = [{"env": i, "group": g, "fell": bool(self.fell[i]), "steps": int(self.err_n[i]),
                     "mean_error_mps": mean[i] if int(self.err_n[i]) else None, "ended": bool(self.done[i])}
                    for i, g in enumerate(self.env_group)]
        cfg = env.cfg
        ck = self.checkpoints[-1] if self.checkpoints else None
        phys = getattr(getattr(env, "sim", None), "physics_manager", None)
        raw = {
            "version": 1, "mode": self.mode, "profile": PROFILE, "profile_record": self.profile_record,
            "task": next((self.play_args[i + 1] for i, a in enumerate(self.play_args[:-1]) if a == "--task"), None),
            "checkpoint": ck, "checkpoint_sha256": ck.get("sha256") if ck else None,
            "checkpoints_loaded": self.checkpoints,
            "command_set_sha256": self.command_set["sha256"], "commands": {**command_sets.summary(
                self.command_set, self.command_source), **self.cmd_info},
            "num_envs": env.num_envs, "episodes_per_group": self.episodes, "episode_steps": self.max_steps,
            "joint_names": list(getattr(env.scene["robot"], "joint_names", None) or []),
            "body_names": list(getattr(env.scene["robot"], "body_names", None) or []),
            "steps_run": self.steps, "all_ended": bool(self.done.all()),
            "step_dt": float(env.step_dt), "sim_dt": float(cfg.sim.dt), "decimation": int(cfg.decimation),
            "seed": getattr(cfg, "seed", None), "overrides": _overrides(self.play_args), "play_args": self.play_args,
            "env_cfg_sha256": _digest(_to_dict(cfg)), "env_cfg_digest_excludes": list(EXCLUDED_FROM_CFG_DIGEST),
            "physics": {"manager": getattr(phys, "__name__", str(phys)), "solver_cfg": _to_dict(cfg.sim.physics),
                        "use_newton_actuators": getattr(cfg.sim, "use_newton_actuators", None),
                        "versions": _versions()},
            "envelope": self.envelope,
            "grid": (robustness.grid_definition(self.envelope) if self.mode == "sweep"
                     else robustness.joint_definition() if self.mode == "joint" else None),
            "groups": self.groups, "conditions": self.applied,
            "training_ranges": training_ranges_near(ck["path"]) if ck and ck.get("path") else None,
            "episodes": episodes,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(raw, f, indent=1)
        nominal = [e for e in episodes if e["group"] == 0] if self.mode != "joint" else []
        print(f"\n[sim2gate] {self.mode}: {len(episodes)} episodes in {len(self.groups)} group(s), "
              f"{sum(e['fell'] for e in episodes)} fell; nominal: {sum(e['fell'] for e in nominal)} of "
              f"{len(nominal)} fell, mean error "
              f"{sum(e['mean_error_mps'] or 0 for e in nominal) / max(1, len(nominal)):.3f} m/s")
        print(f"[sim2gate] raw episodes: {self.out_path}")
        self.raw = raw
        return raw


def isaaclab_api():
    """The Isaac Lab pieces apply_conditions uses (replaced by fakes in tests)."""
    import types
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    def friction_readback(robot):
        import warp as wp
        return wp.to_torch(robot.root_view.get_material_properties())[:, 0, 0]
    return types.SimpleNamespace(
        material=lambda env, params: mdp.randomize_rigid_body_material(
            EventTermCfg(func=mdp.randomize_rigid_body_material, mode="startup", params=params), env),
        mass=lambda env, params: mdp.randomize_rigid_body_mass(
            EventTermCfg(func=mdp.randomize_rigid_body_mass, mode="startup", params=params), env),
        entity=SceneEntityCfg, friction_readback=friction_readback)


def tracking_error(env):
    term = env.command_manager.get_term("base_velocity")
    v = env.scene["robot"].data.root_lin_vel_b
    v = v.torch if hasattr(v, "torch") else v
    return (term.vel_command_b[:, :2] - v[:, :2]).norm(dim=-1)


def apply_conditions(env, cond, group, groups, api=None):
    """Set each environment's physical condition. cond: {param: [N] tensor}. Returns what was applied."""
    api = api or isaaclab_api()
    SceneEntityCfg = api.entity
    rec = {"varied": sorted({g["param"] for g in groups if g["param"]}), "readback": {}}
    robot = env.scene["robot"]
    nominal = robustness.nominal_condition()

    def ids_where(mask):
        # int32, as Isaac Lab's event manager passes them: the PhysX material setter converts them to warp int32
        return mask.nonzero().flatten().to(torch.int32)

    # friction: one material term per value (PhysX samples its material buckets when the term is built)
    fr = cond["ground_friction"]
    for v in sorted(set(fr.tolist())):
        if abs(v - nominal["ground_friction"]) < 1e-6:
            continue
        ids = ids_where((fr - v).abs() < 1e-6)
        sd = (v, v)
        dd = (v * robustness.DYNAMIC_FRICTION_RATIO,) * 2
        acfg = SceneEntityCfg("robot", body_names=".*")
        acfg.resolve(env.scene)
        params = {"asset_cfg": acfg, "static_friction_range": sd, "dynamic_friction_range": dd,
                  "restitution_range": (0.0, 0.0), "num_buckets": 1}
        term = api.material(env, params)
        term(env, ids, **params)
    # added base mass
    ms = cond["added_base_mass_kg"]
    if bool((ms != 0).any()):
        acfg = SceneEntityCfg("robot", body_names="base")
        acfg.resolve(env.scene)
        params = {"asset_cfg": acfg, "mass_distribution_params": (0.0, 0.0), "operation": "add",
                  "distribution": "uniform"}
        term = api.mass(env, params)
        before = _t(robot.data.body_mass).clone()
        for v in sorted(set(ms.tolist())):
            if v == 0:
                continue
            ids = ids_where((ms - v).abs() < 1e-6)
            term(env, ids, acfg, (v, v), "add", "uniform")
        after = _t(robot.data.body_mass)
        b = acfg.body_ids[0] if isinstance(acfg.body_ids, (list, tuple)) else 0
        rec["readback"]["added_base_mass_kg"] = _per_group((after[:, b] - before[:, b]).float(), group, groups)
    # motor strength and latency, in the motor model
    sim_dt = float(env.cfg.sim.dt)
    lat = cond["action_latency_ms"]
    lag = torch.round(lat / (sim_dt * 1000.0)).long()
    if bool(((lag.float() * sim_dt * 1000.0 - lat).abs() > 1e-3).any()):
        raise RuntimeError(f"latency values {sorted(set(lat.tolist()))} ms are not whole physics steps of "
                           f"{sim_dt * 1000:g} ms")
    if int(lag.max()) > MAX_LAG_STEPS:
        raise RuntimeError(f"latency needs {int(lag.max())} physics steps, the motor model buffers {MAX_LAG_STEPS}")
    motors = [a for a in robot.actuators.values() if hasattr(a, "sim2gate_set")]
    if not motors:
        raise RuntimeError("no swept DC motor found; was the nominal profile applied?")
    for a in motors:
        a.sim2gate_set(strength=cond["motor_strength_scale"], lag=lag)
    rec["readback"]["action_latency_ms"] = _per_group(motors[0].sim2gate_lag.double() * sim_dt * 1000.0, group, groups)
    rec["readback"]["motor_strength_scale"] = _per_group(motors[0].sim2gate_strength.flatten(), group, groups)
    # observation noise: each noise term's range scaled per environment
    scale = cond["observation_noise_scale"].reshape(-1, 1).float()      # observations are float32
    noisy = []
    om = env.observation_manager
    for gname, cfgs in om._group_obs_term_cfgs.items():
        if gname != "policy":
            continue
        for name, tc in zip(om._group_obs_term_names[gname], cfgs):
            nz = getattr(tc, "noise", None)
            if nz is None:
                continue
            if hasattr(nz, "n_min") and hasattr(nz, "n_max"):
                nz.n_min = _as_t(nz.n_min, env.device) * scale
                nz.n_max = _as_t(nz.n_max, env.device) * scale
            elif hasattr(nz, "std"):
                nz.std = _as_t(nz.std, env.device) * scale
            else:
                raise RuntimeError(f"observation term {name!r} has a noise model the sweep cannot scale: {type(nz)}")
            noisy.append(name)
    if not noisy:
        raise RuntimeError("the policy observations have no noise terms; is observation corruption on?")
    rec["noise_terms"] = noisy
    rec["readback"]["observation_noise_scale"] = _per_group(scale.flatten(), group, groups)
    # friction readback (PhysX): static friction of the first shape of each group's first env
    try:
        mu = api.friction_readback(robot)
        rec["readback"]["ground_friction"] = _per_group(mu.float().to(group.device), group, groups)
    except Exception as e:                  # recorded, not fatal: the term itself is Isaac Lab's
        rec["readback"]["ground_friction"] = f"not read back: {type(e).__name__}: {e}"
    return rec


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def _as_t(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(float(x), device=device)


def _per_group(values, group, groups):
    """{group: [min, max]} of an env-wise value, in one transfer (the joint sweep has one group per env)."""
    acc = {}
    for g, v in zip(group.flatten().tolist(), values.flatten().tolist()):
        lo_hi = acc.setdefault(g, [v, v])
        lo_hi[0], lo_hi[1] = min(lo_hi[0], v), max(lo_hi[1], v)
    return {str(g["group"]): ([round(float(acc[g["group"]][0]), 6), round(float(acc[g["group"]][1]), 6)]
                              if g["group"] in acc else None) for g in groups}


# ---- training ranges, for the report ---------------------------------------------------------------------------

def training_ranges(env_yaml_text):
    """What the policy's own training run randomized, from its saved params/env.yaml (best effort)."""
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML is not installed"}

    class _L(yaml.SafeLoader):
        pass
    # Isaac Lab's dump_yaml writes tuples as !!python/tuple: read those as lists (they hold every range we report);
    # any other python-specific tag is ignored (None)
    _L.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
    _L.add_multi_constructor("", lambda loader, suffix, node: None)
    try:
        d = yaml.load(env_yaml_text, Loader=_L) or {}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    ev = d.get("events") or {}
    out = {}
    pm = (ev.get("physics_material") or {}).get("params") or {}
    out["ground_friction"] = {"static_friction_range": pm.get("static_friction_range")} if pm else "not randomized"
    bm = (ev.get("add_base_mass") or {}).get("params") or {}
    out["added_base_mass_kg"] = ({"operation": bm.get("operation"), "params": bm.get("mass_distribution_params"),
                                  "distribution": bm.get("distribution")} if bm else "not randomized")
    acts = (((d.get("scene") or {}).get("robot") or {}).get("actuators") or {})
    delays = {k: [a.get("min_delay"), a.get("max_delay")] for k, a in acts.items()
              if isinstance(a, dict) and ("min_delay" in a or "max_delay" in a)}
    out["action_latency_ms"] = delays or "no delay in the actuator model"
    pay = (ev.get("add_payload") or {}).get("params") or {}
    if pay:
        out["added_base_mass_kg"] = {"stock_scale": out["added_base_mass_kg"],
                                     "payload": {"operation": pay.get("operation"),
                                                 "params": pay.get("mass_distribution_params"),
                                                 "distribution": pay.get("distribution")}}
    strengths = {k: [a.get("min_strength"), a.get("max_strength")] for k, a in acts.items()
                 if isinstance(a, dict) and a.get("min_strength") is not None
                 and a.get("min_strength") != a.get("max_strength")}
    out["motor_strength_scale"] = (strengths if strengths else
                                   "not randomized" if not ev.get("actuator_gains") else ev.get("actuator_gains"))
    pol = ((d.get("observations") or {}).get("policy") or {})
    out["observation_noise_scale"] = {"enable_corruption": pol.get("enable_corruption")}
    return out


def training_ranges_near(checkpoint_path):
    p = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "params", "env.yaml")
    if not os.path.exists(p):
        return None
    try:
        return training_ranges(open(p).read())
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---- installation --------------------------------------------------------------------------------------------------

def install(rec, wrapper_cls=None, runner_cls=None, gym_module=None):
    """Patch gymnasium.make (to apply the profile to the env config before the env is built), the RSL-RL wrapper's
    step and the runner's load. Returns undo().

    The env config is reached through gymnasium.make(task, cfg=...), which is how Isaac Lab's play script builds the
    environment. Nothing from isaaclab.envs is imported here: on the Isaac Sim (Kit) path that would load Omniverse
    modules before Isaac Sim has started, and Isaac Sim then fails to start."""
    if wrapper_cls is None:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as wrapper_cls
    if gym_module is None:
        import gymnasium as gym_module
    if runner_cls is None:
        try:
            from rsl_rl.runners import OnPolicyRunner as runner_cls
        except ImportError:
            runner_cls = None
    undo = []
    orig_make = gym_module.make

    def make(*a, **k):
        cfg = k.get("cfg")
        if cfg is None:
            raise RuntimeError("gymnasium.make was called without cfg=; cannot apply the nominal profile")
        rec.on_env_cfg(cfg)
        return orig_make(*a, **k)
    gym_module.make = make
    undo.append(lambda: setattr(gym_module, "make", orig_make))
    orig_step = wrapper_cls.step

    def step(self, actions):
        return rec.step(self, orig_step, actions)
    wrapper_cls.step = step
    undo.append(lambda: setattr(wrapper_cls, "step", orig_step))
    if runner_cls is not None:
        orig_load = runner_cls.load

        def load(self, path, *a, **k):
            rec.record_checkpoint(path)
            return orig_load(self, path, *a, **k)
        runner_cls.load = load
        undo.append(lambda: setattr(runner_cls, "load", orig_load))
    return lambda: [u() for u in reversed(undo)]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, rest = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    p = argparse.ArgumentParser(prog="python -m sim2gate.checks.isaaclab_conditions",
                                description="Record a policy's episodes under controlled conditions (checks 1, 5).")
    p.add_argument("--mode", choices=("sweep", "nominal", "joint"), required=True,
                   help="sweep: check 1's whole grid; nominal: the nominal condition only (check 5); joint: all five "
                        "conditions drawn together per episode (report only)")
    p.add_argument("--commands", required=True, help="fixed command set JSON, or 'mvp' for the MVP's Go2 set")
    p.add_argument("--out", required=True)
    p.add_argument("--envelope", choices=sorted(robustness.ENVELOPES), default="v1",
                   help="operating envelope swept by check 1 (v1: the MVP's, frozen; v2: the v2 plan's)")
    p.add_argument("--repeat", type=int, default=1,
                   help="episodes per condition = repeat x the command set's size (v2 nominal: 4, i.e. 256)")
    a = p.parse_args(ours)
    if a.repeat < 1:
        p.error("--repeat must be at least 1")
    if not any(x == f or x.startswith(f + "=") for x in rest for f in ("--checkpoint", "--load_run")):
        p.error("name the policy: pass --checkpoint <model.pt> after --")
    if any(x == "--num_envs" or x.startswith("--num_envs=") for x in rest):
        p.error("--num_envs is set by the mode; do not pass it")
    if a.commands == "mvp":
        from ..mvp.baselines import GO2_COMMANDS
        a.commands = GO2_COMMANDS
    doc = command_sets.load_command_set(a.commands)
    rec = ConditionRecorder(a.mode, a.out, doc, os.path.abspath(a.commands), rest,
                            episodes=len(doc["commands"]) * a.repeat, envelope=a.envelope)
    if os.path.exists(a.out):
        os.remove(a.out)
    install(rec)
    sys.argv = ["play_rsl_rl"] + rest + ["--num_envs", str(rec.num_envs)]
    try:
        runpy.run_module("isaaclab_rl.entrypoints.backends.play_rsl_rl", run_name="__main__")
    except KeyboardInterrupt:
        pass
    return 0 if os.path.exists(a.out) else 1


if __name__ == "__main__":
    sys.exit(main())
