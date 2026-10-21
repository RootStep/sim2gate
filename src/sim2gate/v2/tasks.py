"""Sim2Gate v2 task variants. P0 v5: the v2 clean robot (v2 plan, "Layer 1: operating envelope").

    P0 v5 = P0 v4 trained across the whole v2 operating envelope (VJ's decision, Oct 19, 2026: "full envelope"):
            - added payload on the base, uniform 0-8 kg per environment (startup; Isaac Lab's own mass event, "add"),
              on top of the stock base-mass scale;
            - ground friction static 0.35-1.2 (dynamic 0.2625-0.9, the task's 0.75 ratio, never above static);
            - motor strength (output torque scale) 0.8-1.0, drawn per episode, on P0 v4's ramp (full strength before
              round 500, the full range from round 800), like the 0-30 ms command delay;
            - everything else as P0 v4 (foot-slip penalty, 2000 rounds). Sensor noise stays at the task default.
    Evaluation (check 1's nominal profile) removes the payload and stock mass events and replaces the motor, so a
    clean robot is always measured at nominal, then swept.

    P0 v5b = P0 v5 with the payload phased in like the delay and motor strength (VJ, Oct 21, 2026, after the P0 v5
             pilot: seed 1 learned to stand rather than walk, speed-tracking reward 0.23 vs P0 v4's 1.36). The payload
             is drawn per episode at each reset, uniform 0 to an upper bound that is 0 before round 500 and rises to
             8 kg by round 800 (as the delay); play mode draws from the full 0-8 kg.

The v1 test set (sim2gate.mvp.tasks) is unchanged. Isaac Lab loads these through --external_callback
sim2gate.v2.tasks.register, which also registers the MVP tasks.
"""
P0_TASK_V5 = "Sim2Gate-Velocity-Flat-Go2-P0-v5"
P0_V5_STATIC_FRICTION = (0.35, 1.2)
P0_V5_DYNAMIC_FRICTION = (0.2625, 0.9)
P0_V5_PAYLOAD_KG = (0.0, 8.0)
P0_V5_MOTOR_STRENGTH = (0.8, 1.0)
P0_V5_ITERATIONS = 2000
P0_TASK_V5B = "Sim2Gate-Velocity-Flat-Go2-P0-v5b"
ROUND_ENV_STEPS = 24                               # RSL-RL rollout length: environment steps per training round
P0_V5B_PAYLOAD_RAMP_ROUNDS = (500, 800)            # as P0 v4's delay ramp (sim2gate.mvp.tasks.P0_V4_DELAY_RAMP_ROUNDS)
P0_TASK_CURRENT = P0_TASK_V5B                      # the P0 phase 1 trains now
CALLBACK = ["--external_callback", "sim2gate.v2.tasks.register"]

VARIANTS = {
    P0_TASK_V5: {
        "policy": "P0", "version": 5, "base_task": "Sim2Gate-Velocity-Flat-Go2-P0-v4",
        "changes": [
            {"what": "events.add_payload (new)", "func": "isaaclab_tasks.core.velocity.mdp.randomize_rigid_body_mass",
             "mode": "startup", "body": "base", "operation": "add", "distribution": "uniform",
             "kg": list(P0_V5_PAYLOAD_KG)},
            {"what": "events.physics_material ranges", "static_friction_range": list(P0_V5_STATIC_FRICTION),
             "dynamic_friction_range": list(P0_V5_DYNAMIC_FRICTION), "make_consistent": True},
            {"what": "actuators.base_legs motor strength per episode", "range": list(P0_V5_MOTOR_STRENGTH),
             "ramp": "P0 v4's delay ramp (rounds 500-800); play mode: full range"}],
        "max_iterations": P0_V5_ITERATIONS, "experiment_name": "sim2gate_go2_flat_p0_v5",
        "status": "v2 pilot (phase 1): frozen only after the pilot passes",
        "env_class": "Go2FlatP0EnvCfgV5", "agent_class": "Go2FlatP0PPORunnerCfgV5"},
}
VARIANTS[P0_TASK_V5]["status"] = "superseded by v5b: the pilot's seed 1 learned to stand, not walk (Oct 21, 2026)"
VARIANTS[P0_TASK_V5B] = {
    **VARIANTS[P0_TASK_V5], "version": "5b", "base_task": P0_TASK_V5,
    "changes": VARIANTS[P0_TASK_V5]["changes"][1:] + [
        {"what": "events.add_payload: per episode, ramped", "class": "sim2gate.v2.tasks:RampedPayload",
         "mode": "reset", "body": "base", "operation": "add", "distribution": "uniform",
         "kg": list(P0_V5_PAYLOAD_KG), "ramp_rounds": list(P0_V5B_PAYLOAD_RAMP_ROUNDS),
         "ramp_env_steps": [r * ROUND_ENV_STEPS for r in P0_V5B_PAYLOAD_RAMP_ROUNDS],
         "play_mode": "full 0-8 kg, no ramp"}],
    "experiment_name": "sim2gate_go2_flat_p0_v5b",
    "status": "v2 pilot (phase 1): frozen only after the pilot passes",
    "env_class": "Go2FlatP0EnvCfgV5b", "agent_class": "Go2FlatP0PPORunnerCfgV5b"}


def payload_upper_kg(env_step, max_kg, ramp_env_steps):
    """The largest payload drawn at this environment step: 0 before the ramp, max_kg after it, linear between."""
    s0, s1 = ramp_env_steps
    if s1 > s0:
        frac = min(1.0, max(0.0, (env_step - s0) / (s1 - s0)))
    else:
        frac = 1.0 if env_step >= s0 else 0.0
    return float(max_kg) * frac


_PAYLOAD_CLS = None


def ramped_payload_class(base_cls=None, term_cfg_cls=None):
    """RampedPayload: an Isaac Lab event term (mode "reset") that re-draws each reset environment's added base mass,
    uniform in [0, payload_upper_kg(...)], through Isaac Lab's own randomize_rigid_body_mass ("add" on the base's
    mass at its first call, which already includes the stock start-up mass scale). The arguments replace Isaac Lab's
    classes in tests."""
    global _PAYLOAD_CLS
    if base_cls is None and _PAYLOAD_CLS is not None:
        return _PAYLOAD_CLS
    if base_cls is None:
        from isaaclab.managers import EventTermCfg, ManagerTermBase
        from isaaclab_tasks.core.velocity import mdp
        base_cls, term_cfg_cls, inner_cls = ManagerTermBase, EventTermCfg, mdp.randomize_rigid_body_mass
    else:
        inner_cls = base_cls.inner_cls

    class RampedPayload(base_cls):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            p = cfg.params
            inner_cfg = term_cfg_cls(func=inner_cls, mode="reset",
                                     params={"asset_cfg": p["asset_cfg"], "mass_distribution_params": (0.0, 0.0),
                                             "operation": "add", "distribution": "uniform"})
            self._inner = inner_cls(inner_cfg, env)
            self.last_upper_kg = None

        def __call__(self, env, env_ids, asset_cfg, mass_distribution_params, ramp_env_steps, operation="add",
                     distribution="uniform"):
            hi = payload_upper_kg(int(getattr(env, "common_step_counter", 0)), mass_distribution_params[1],
                                  ramp_env_steps)
            self.last_upper_kg = hi
            # always called, so an environment's payload is reset to 0 at each episode before the ramp starts
            return self._inner(env, env_ids, asset_cfg, (float(mass_distribution_params[0]), hi), "add", "uniform")

    RampedPayload.__qualname__ = "RampedPayload"     # resolvable as sim2gate.v2.tasks:RampedPayload in saved configs
    RampedPayload.__module__ = __name__
    if base_cls.__module__.startswith("isaaclab"):
        _PAYLOAD_CLS = RampedPayload
    return RampedPayload


def _p0v5b_env_cfg():
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.utils import configclass
    base = __getattr__("Go2FlatP0EnvCfgV5")
    ramp = tuple(r * ROUND_ENV_STEPS for r in P0_V5B_PAYLOAD_RAMP_ROUNDS)

    @configclass
    class Go2FlatP0EnvCfgV5b(base):
        def __post_init__(self):
            super().__post_init__()
            self.events.add_payload = EventTermCfg(
                func=ramped_payload_class(), mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                        "mass_distribution_params": P0_V5_PAYLOAD_KG, "ramp_env_steps": ramp,
                        "operation": "add", "distribution": "uniform"})

        def play_mode(self):
            super().play_mode()
            if getattr(self.events, "add_payload", None) is not None:
                self.events.add_payload.params["ramp_env_steps"] = (0, 0)      # evaluation: the full 0-8 kg

    return Go2FlatP0EnvCfgV5b


def _p0v5b_agent_cfg():
    from isaaclab.utils import configclass
    base = __getattr__("Go2FlatP0PPORunnerCfgV5")
    v = VARIANTS[P0_TASK_V5B]

    @configclass
    class Go2FlatP0PPORunnerCfgV5b(base):
        def __post_init__(self):
            super().__post_init__()
            self.experiment_name = v["experiment_name"]
            self.max_iterations = v["max_iterations"]

    return Go2FlatP0PPORunnerCfgV5b


def _p0v5_env_cfg():
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.core.velocity import mdp            # as the stock add_base_mass event
    from ..mvp import tasks
    base = tasks._p0_env_cfg(4)

    @configclass
    class Go2FlatP0EnvCfgV5(base):
        def __post_init__(self):
            super().__post_init__()
            pm = self.events.physics_material.params
            pm["static_friction_range"] = P0_V5_STATIC_FRICTION
            pm["dynamic_friction_range"] = P0_V5_DYNAMIC_FRICTION
            pm["make_consistent"] = True
            self.events.add_payload = EventTermCfg(
                func=mdp.randomize_rigid_body_mass, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                        "mass_distribution_params": P0_V5_PAYLOAD_KG, "operation": "add", "distribution": "uniform"})
            legs = self.scene.robot.actuators["base_legs"]
            if not hasattr(legs, "min_strength"):
                raise RuntimeError("P0 v4's delayed DC motor was expected on base_legs")
            legs.min_strength, legs.max_strength = P0_V5_MOTOR_STRENGTH

    return Go2FlatP0EnvCfgV5


def _p0v5_agent_cfg():
    from isaaclab.utils import configclass
    from ..mvp import tasks
    base = tasks._p0_agent_cfg(tasks.P0_TASK_V4)
    v = VARIANTS[P0_TASK_V5]

    @configclass
    class Go2FlatP0PPORunnerCfgV5(base):
        def __post_init__(self):
            super().__post_init__()
            self.experiment_name = v["experiment_name"]
            self.max_iterations = v["max_iterations"]

    return Go2FlatP0PPORunnerCfgV5


_LAZY = {"Go2FlatP0EnvCfgV5": _p0v5_env_cfg, "Go2FlatP0PPORunnerCfgV5": _p0v5_agent_cfg,
         "Go2FlatP0EnvCfgV5b": _p0v5b_env_cfg, "Go2FlatP0PPORunnerCfgV5b": _p0v5b_agent_cfg}
_built = {}


def __getattr__(name):
    if name == "RampedPayload":
        return ramped_payload_class()
    if name in _LAZY:
        if name not in _built:
            _built[name] = _LAZY[name]()
        return _built[name]
    raise AttributeError(name)


def register():
    """Isaac Lab --external_callback: the MVP tasks and the v2 tasks."""
    import gymnasium as gym
    from ..mvp import tasks
    tasks.register()
    for task, v in VARIANTS.items():
        if task not in gym.registry:
            gym.register(id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                         kwargs={"env_cfg_entry_point": f"{__name__}:{v['env_class']}",
                                 "rsl_rl_cfg_entry_point": f"{__name__}:{v['agent_class']}",
                                 "default_agent": "rsl_rl", "sim2gate_variant": v})
    return None
