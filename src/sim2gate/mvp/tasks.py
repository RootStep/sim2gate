"""Registered, versioned task variants for the MVP test set (docs/mvp-plan, change C4).

    P0 v1 = Isaac Lab's Go2 flat velocity task (Isaac-Velocity-Flat-UnitreeGo2) + Isaac Lab's own foot-slip penalty
            (mdp.feet_slide on the four feet, weight P0_FEET_SLIDE_WEIGHT), the task's 300 training iterations
            (change C4; superseded by v2: seed 2 fell in 15 of 64 episodes)
    P0 v2 = P0 v1 trained for 1000 iterations instead of 300; nothing else changes (change C5)
    P0 v3 = P0 v2 + training randomization the plan assumed the clean baseline has (change C6): ground friction
            static 0.4-1.2 (dynamic 0.3-0.9, never above static) and a command delay of 0-30 ms (0-6 physics steps,
            drawn per episode) in the DC motor model
    P0 v4 = P0 v3 with the delay introduced after the robot has learned to walk (change C7): no delay for the first
            500 training rounds, the largest delay rising to 30 ms by round 800, the full 0-30 ms from then on;
            2000 rounds. Evaluation (play mode) always draws from the full 0-30 ms

The planted-flaw policies P1-P5 (test set v1, docs/mvp-plan/test-set.md) are each P0 v4 with exactly one change:
    P1 narrow randomization: friction static 0.78-0.82 (dynamic 0.585-0.615) and base mass scale 1/1.01-1.01
    P2 no latency: command delay 0 throughout (the same delay motor, range 0-0)
    P3 sliding exploit: low ground friction, static 0.2-0.4 (dynamic 0.15-0.3), and no foot-slip penalty
    P4 jittery: no action-rate penalty
    P5 engine overfit: a soft, PhysX-only ground contact (compliant spring in acceleration units, which Newton does
       not reproduce)
Each is evaluated in its own task (play mode), like any user's policy in the simulator it was trained in.

The stock task has no foot-slip penalty, and the plan defines P3 as "foot-slip penalty removed" (plus low ground
friction), so the clean baseline must have one. The weight -0.25 is a proposed starting value, taken from Isaac
Lab's H1 configuration; it is not a validated Go2 setting. Whether P0 v1 is an acceptable clean baseline is decided
by the criteria in sim2gate.mvp.baselines.BASELINE_CRITERIA, fixed before training.

Isaac Lab loads these through its --external_callback hook (train and play):

    --external_callback sim2gate.mvp.tasks.register

Nothing is imported from Isaac Lab until the task is built, so this module imports without it. Changing a
variant means a new version (new task id); a registered version never changes.
"""

P0_TASK_V1 = "Sim2Gate-Velocity-Flat-Go2-P0-v1"
P0_TASK_V2 = "Sim2Gate-Velocity-Flat-Go2-P0-v2"
P0_TASK_V3 = "Sim2Gate-Velocity-Flat-Go2-P0-v3"
P0_TASK_V4 = "Sim2Gate-Velocity-Flat-Go2-P0-v4"
P0_TASK = P0_TASK_V4                     # the current P0
P0_FEET_SLIDE_WEIGHT = -0.25
P0_V2_ITERATIONS = 1000
P0_V3_STATIC_FRICTION = (0.4, 1.2)       # the plan's plausible real range (check 1)
P0_V3_DYNAMIC_FRICTION = (0.3, 0.9)      # the task's dynamic/static ratio (0.75) at both ends; make_consistent caps it
P0_V3_DELAY_STEPS = (0, 6)               # 0-30 ms at the task's 5 ms physics step (the plan's plausible latency)
PHYSICS_STEPS_PER_ROUND = 24 * 4         # RSL-RL rollout (num_steps_per_env 24) x decimation 4
P0_V4_DELAY_RAMP_ROUNDS = (500, 800)     # C7: no delay before round 500, full range from round 800
P0_V4_ITERATIONS = 2000

VARIANTS = {
    P0_TASK_V1: {
        "policy": "P0", "version": 1, "base_task": "Isaac-Velocity-Flat-UnitreeGo2",
        "changes": [{"what": "add reward term rewards.feet_slide", "func": "isaaclab_tasks.core.velocity.mdp.feet_slide",
                     "weight": P0_FEET_SLIDE_WEIGHT, "feet": ".*_foot", "sensor": "contact_forces",
                     "weight_status": "proposed starting value (Isaac Lab H1), not validated for the Go2"}],
        "max_iterations": 300, "experiment_name": "sim2gate_go2_flat_p0_v1", "status": "superseded by v2 (C5)",
    },
    P0_TASK_V2: {
        "policy": "P0", "version": 2, "base_task": "Isaac-Velocity-Flat-UnitreeGo2",
        "changes": [{"what": "add reward term rewards.feet_slide", "func": "isaaclab_tasks.core.velocity.mdp.feet_slide",
                     "weight": P0_FEET_SLIDE_WEIGHT, "feet": ".*_foot", "sensor": "contact_forces",
                     "weight_status": "proposed starting value (Isaac Lab H1), not validated for the Go2"},
                    {"what": "training iterations 300 -> 1000", "max_iterations": P0_V2_ITERATIONS}],
        "max_iterations": P0_V2_ITERATIONS, "experiment_name": "sim2gate_go2_flat_p0_v2", "status": "superseded by v3 (C6)",
    },
}
VARIANTS[P0_TASK_V3] = {
    "policy": "P0", "version": 3, "base_task": "Isaac-Velocity-Flat-UnitreeGo2",
    "changes": VARIANTS[P0_TASK_V2]["changes"] + [
        {"what": "events.physics_material ranges", "static_friction_range": list(P0_V3_STATIC_FRICTION),
         "dynamic_friction_range": list(P0_V3_DYNAMIC_FRICTION), "make_consistent": True},
        {"what": "actuators.base_legs: DC motor with a per-episode command delay",
         "class": "sim2gate.mvp.actuators.DelayedDCMotor", "delay_physics_steps": list(P0_V3_DELAY_STEPS),
         "delay_ms": [5 * P0_V3_DELAY_STEPS[0], 5 * P0_V3_DELAY_STEPS[1]]},
        {"what": "sim.use_newton_actuators = False", "why": "Isaac Lab rejects a custom motor under Newton-native "
         "motors, also on PhysX, where the setting has no effect (Oct 15)"}],
    "max_iterations": P0_V2_ITERATIONS, "experiment_name": "sim2gate_go2_flat_p0_v3", "status": "superseded by v4 (C7)",
}
VARIANTS[P0_TASK_V4] = {
    "policy": "P0", "version": 4, "base_task": "Isaac-Velocity-Flat-UnitreeGo2",
    "changes": VARIANTS[P0_TASK_V3]["changes"] + [
        {"what": "delay ramp in training", "no_delay_before_round": P0_V4_DELAY_RAMP_ROUNDS[0],
         "full_delay_from_round": P0_V4_DELAY_RAMP_ROUNDS[1],
         "physics_steps": [r * PHYSICS_STEPS_PER_ROUND for r in P0_V4_DELAY_RAMP_ROUNDS],
         "evaluation": "play mode: full 0-30 ms range, no ramp"},
        {"what": "training rounds 1000 -> 2000", "max_iterations": P0_V4_ITERATIONS}],
    "max_iterations": P0_V4_ITERATIONS, "experiment_name": "sim2gate_go2_flat_p0_v4", "status": "current",
}

# ---- test set v1: the planted-flaw policies P1-P5 (each = P0 v4 + one change) ------------------------------------
TEST_SET_VERSION = 1
P1_STATIC_FRICTION = (0.78, 0.82)        # "near-zero range" around the task's nominal 0.8
P1_DYNAMIC_FRICTION = (0.585, 0.615)     # the same 0.75 dynamic/static ratio
P1_MASS_SCALE = (1 / 1.01, 1.01)         # the stock task's base-mass scale (1/1.25-1.25, log-uniform) narrowed to 1%
P2_DELAY_STEPS = (0, 0)
P3_STATIC_FRICTION = (0.2, 0.4)          # below the plausible real range (0.4-1.2)
P3_DYNAMIC_FRICTION = (0.15, 0.3)
P5_CONTACT_STIFFNESS = 1000.0            # PhysX compliant contact, acceleration units (1/s^2); NOT FINAL: the Oct 16
                                         # stand test sank the robot 8.7 cm on PhysX (intended ~1 cm); a pilot picks the value
P5_CONTACT_DAMPING = 60.0                # about critical (2 x sqrt(1000) = 63)
# Pilot only (Oct 16, before the freeze and before any training): a stiffness to stand-test instead of the value above,
# with damping 2 x sqrt(stiffness). Removed once the value is chosen; the blinded run never sets it.
P5_PILOT_STIFFNESS_ENV = "SIM2GATE_P5_PILOT_STIFFNESS"


def p5_contact():
    """(stiffness, damping) for P5's ground: the defined values, or the pilot's if SIM2GATE_P5_PILOT_STIFFNESS is set."""
    import math
    import os
    k = os.environ.get(P5_PILOT_STIFFNESS_ENV)
    if not k:
        return P5_CONTACT_STIFFNESS, P5_CONTACT_DAMPING
    k = float(k)
    return k, 2.0 * math.sqrt(k)
PLANTED = {
    "P1": {"flaw": "narrow randomization", "target_check": 1,
           "changes": [{"what": "events.physics_material ranges", "static_friction_range": list(P1_STATIC_FRICTION),
                        "dynamic_friction_range": list(P1_DYNAMIC_FRICTION), "make_consistent": True},
                       {"what": "events.add_base_mass scale range", "mass_distribution_params": list(P1_MASS_SCALE),
                        "operation": "scale", "distribution": "log_uniform"}]},
    "P2": {"flaw": "no latency", "target_check": 1,
           "changes": [{"what": "actuators.base_legs command delay", "delay_physics_steps": list(P2_DELAY_STEPS)}]},
    "P3": {"flaw": "sliding exploit", "target_check": 2,
           "changes": [{"what": "events.physics_material ranges", "static_friction_range": list(P3_STATIC_FRICTION),
                        "dynamic_friction_range": list(P3_DYNAMIC_FRICTION), "make_consistent": True},
                       {"what": "remove reward term rewards.feet_slide"}]},
    "P4": {"flaw": "jittery", "target_check": 2,
           "changes": [{"what": "remove reward term rewards.action_rate_l2"}]},
    "P5": {"flaw": "engine overfit", "target_check": 5,
           "changes": [{"what": "ground material: PhysX compliant contact, acceleration spring",
                        "compliant_contact_stiffness": P5_CONTACT_STIFFNESS,
                        "compliant_contact_damping": P5_CONTACT_DAMPING,
                        "compliant_contact_acceleration_spring": True,
                        "why": "Isaac Lab: the acceleration-spring flag is PhysX-only, not consumed by Newton, which "
                               "reads the same stiffness and damping as a force spring"}]},
}
PLANTED_TASKS = {}
for _pol, _d in PLANTED.items():
    _task = f"Sim2Gate-Velocity-Flat-Go2-{_pol}-v{TEST_SET_VERSION}"
    PLANTED_TASKS[_pol] = _task
    VARIANTS[_task] = {
        "policy": _pol, "version": TEST_SET_VERSION, "base_task": P0_TASK_V4, "flaw": _d["flaw"],
        "target_check": _d["target_check"], "changes": _d["changes"],
        "max_iterations": P0_V4_ITERATIONS, "experiment_name": f"sim2gate_go2_flat_{_pol.lower()}_v{TEST_SET_VERSION}",
        "status": "test set v1", "env_class": f"Go2Flat{_pol}EnvCfg", "agent_class": f"Go2Flat{_pol}PPORunnerCfg"}


def _p0_env_cfg(version=1):
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.core.velocity import mdp
    from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

    @configclass
    class Go2FlatP0EnvCfg(UnitreeGo2FlatEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.rewards.feet_slide = RewardTermCfg(
                func=mdp.feet_slide, weight=P0_FEET_SLIDE_WEIGHT,
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                        "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")})
            if version >= 3:
                from .actuators import with_delay
                pm = self.events.physics_material.params
                pm["static_friction_range"] = P0_V3_STATIC_FRICTION
                pm["dynamic_friction_range"] = P0_V3_DYNAMIC_FRICTION
                pm["make_consistent"] = True
                acts = self.scene.robot.actuators
                ramp = tuple(r * PHYSICS_STEPS_PER_ROUND for r in P0_V4_DELAY_RAMP_ROUNDS) if version >= 4 else (0, 0)
                acts["base_legs"] = with_delay(acts["base_legs"], *P0_V3_DELAY_STEPS, ramp_steps=ramp)
                # The Go2 task asks for Newton-native motors; Isaac Lab validates that against every actuator config,
                # on PhysX too, and rejects a custom motor. On PhysX the setting has no effect (Isaac Lab's own motor
                # model always runs there), and checks 1 and 5 already run with it off.
                self.sim.use_newton_actuators = False

        def play_mode(self):
            super().play_mode()
            act = self.scene.robot.actuators.get("base_legs")
            if act is not None and hasattr(act, "delay_ramp_end_steps"):
                act.delay_ramp_start_steps = 0          # evaluation: the full delay range from the first episode
                act.delay_ramp_end_steps = 0

    Go2FlatP0EnvCfg.__name__ = f"Go2FlatP0EnvCfgV{version}"
    return Go2FlatP0EnvCfg


def _p0_agent_cfg(task):
    from isaaclab.utils import configclass
    from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg
    v = VARIANTS[task]

    @configclass
    class Go2FlatP0PPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
        def __post_init__(self):
            super().__post_init__()
            self.experiment_name = v["experiment_name"]
            self.max_iterations = v["max_iterations"]

    Go2FlatP0PPORunnerCfg.__name__ = f"Go2FlatP0PPORunnerCfgV{v['version']}"
    return Go2FlatP0PPORunnerCfg


def _planted_env_cfg(pol):
    """P0 v4's environment with policy `pol`'s one planted change."""
    from isaaclab.utils import configclass
    base = _p0_env_cfg(4)

    @configclass
    class Go2FlatPlantedEnvCfg(base):
        def __post_init__(self):
            super().__post_init__()
            if pol == "P1":
                pm = self.events.physics_material.params
                pm["static_friction_range"], pm["dynamic_friction_range"] = P1_STATIC_FRICTION, P1_DYNAMIC_FRICTION
                self.events.add_base_mass.params["mass_distribution_params"] = P1_MASS_SCALE
            elif pol == "P2":
                act = self.scene.robot.actuators["base_legs"]
                act.min_delay, act.max_delay = P2_DELAY_STEPS
            elif pol == "P3":
                pm = self.events.physics_material.params
                pm["static_friction_range"], pm["dynamic_friction_range"] = P3_STATIC_FRICTION, P3_DYNAMIC_FRICTION
                self.rewards.feet_slide = None
            elif pol == "P4":
                self.rewards.action_rate_l2 = None
            elif pol == "P5":
                try:
                    from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
                except ImportError:                                      # older layout
                    from isaaclab.sim.spawners.materials import PhysxRigidBodyMaterialCfg
                ground = self.scene.terrain.physics_material
                mat = PhysxRigidBodyMaterialCfg(
                    static_friction=ground.static_friction, dynamic_friction=ground.dynamic_friction,
                    friction_combine_mode="multiply", restitution_combine_mode="multiply",
                    compliant_contact_stiffness=p5_contact()[0], compliant_contact_damping=p5_contact()[1],
                    compliant_contact_acceleration_spring=True)
                self.scene.terrain.physics_material = mat
                self.sim.physics_material = mat
            else:
                raise ValueError(f"no planted change for {pol}")

    Go2FlatPlantedEnvCfg.__name__ = f"Go2Flat{pol}EnvCfg"
    return Go2FlatPlantedEnvCfg


# P0 v1 and v2 share the environment (the same penalty) and differ only in the training budget; v3 adds C6's
# friction and delay randomization
_LAZY = {"Go2FlatP0EnvCfgV1": lambda: _p0_env_cfg(1), "Go2FlatP0EnvCfgV2": lambda: _p0_env_cfg(2),
         "Go2FlatP0EnvCfgV3": lambda: _p0_env_cfg(3), "Go2FlatP0EnvCfgV4": lambda: _p0_env_cfg(4),
         "Go2FlatP0PPORunnerCfgV1": lambda: _p0_agent_cfg(P0_TASK_V1),
         "Go2FlatP0PPORunnerCfgV2": lambda: _p0_agent_cfg(P0_TASK_V2),
         "Go2FlatP0PPORunnerCfgV3": lambda: _p0_agent_cfg(P0_TASK_V3),
         "Go2FlatP0PPORunnerCfgV4": lambda: _p0_agent_cfg(P0_TASK_V4)}
for _pol, _task in PLANTED_TASKS.items():
    _LAZY[f"Go2Flat{_pol}EnvCfg"] = (lambda p: (lambda: _planted_env_cfg(p)))(_pol)
    _LAZY[f"Go2Flat{_pol}PPORunnerCfg"] = (lambda t: (lambda: _p0_agent_cfg(t)))(_task)
_built = {}


def __getattr__(name):
    """Build the config classes on first use (by Isaac Lab, through the registered entry points)."""
    if name in _LAZY:
        if name not in _built:
            _built[name] = _LAZY[name]()
        return _built[name]
    raise AttributeError(name)


BLIND_ALIASES_ENV = "SIM2GATE_BLIND_ALIASES"


def _entry(v):
    n = v["version"]
    env = v.get("env_class", f"Go2FlatP0EnvCfgV{n}")
    agent = v.get("agent_class", f"Go2FlatP0PPORunnerCfgV{n}")
    return {"env_cfg_entry_point": f"{__name__}:{env}", "rsl_rl_cfg_entry_point": f"{__name__}:{agent}"}


def task_token(task):
    """An opaque token for a task id (the blinded run's aliases file holds these, not task names)."""
    import hashlib
    return hashlib.sha256(task.encode()).hexdigest()[:16]


def blind_aliases(path=None):
    """{alias task id: real task id} from the blinded run's aliases file (sim2gate.mvp.blinded), or {}. The file
    maps each alias to an opaque token, so reading it does not show which policy an ID is."""
    import json
    import os
    path = path or os.environ.get(BLIND_ALIASES_ENV)
    if not path:
        return {}
    tokens = {task_token(t): t for t in VARIANTS}
    doc = json.load(open(path))
    bad = [t for t in doc.values() if t not in tokens]
    if bad:
        raise ValueError(f"{path}: aliases point at unknown tasks {bad}")
    return {alias: tokens[t] for alias, t in doc.items()}


def register():
    """Isaac Lab --external_callback: register the MVP task variants with Gymnasium, and, during a blinded run, each
    policy's opaque alias (Sim2Gate-Blind-<ID>) for its task, so logs and records name only the blinded ID. Returns
    None, so Isaac Lab keeps every remaining command-line argument."""
    import gymnasium as gym
    for task, v in VARIANTS.items():
        if task not in gym.registry:
            gym.register(id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                         kwargs={**_entry(v), "default_agent": "rsl_rl", "sim2gate_variant": v})
    for alias, task in blind_aliases().items():
        if alias not in gym.registry:
            gym.register(id=alias, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                         kwargs={**_entry(VARIANTS[task]), "default_agent": "rsl_rl"})
    return None
