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


# P0 v1 and v2 share the environment (the same penalty) and differ only in the training budget; v3 adds C6's
# friction and delay randomization
_LAZY = {"Go2FlatP0EnvCfgV1": lambda: _p0_env_cfg(1), "Go2FlatP0EnvCfgV2": lambda: _p0_env_cfg(2),
         "Go2FlatP0EnvCfgV3": lambda: _p0_env_cfg(3), "Go2FlatP0EnvCfgV4": lambda: _p0_env_cfg(4),
         "Go2FlatP0PPORunnerCfgV1": lambda: _p0_agent_cfg(P0_TASK_V1),
         "Go2FlatP0PPORunnerCfgV2": lambda: _p0_agent_cfg(P0_TASK_V2),
         "Go2FlatP0PPORunnerCfgV3": lambda: _p0_agent_cfg(P0_TASK_V3),
         "Go2FlatP0PPORunnerCfgV4": lambda: _p0_agent_cfg(P0_TASK_V4)}
_built = {}


def __getattr__(name):
    """Build the config classes on first use (by Isaac Lab, through the registered entry points)."""
    if name in _LAZY:
        if name not in _built:
            _built[name] = _LAZY[name]()
        return _built[name]
    raise AttributeError(name)


def register():
    """Isaac Lab --external_callback: register the MVP task variants with Gymnasium. Returns None, so Isaac Lab
    keeps every remaining command-line argument."""
    import gymnasium as gym
    for task, v in VARIANTS.items():
        if task not in gym.registry:
            n = v["version"]
            gym.register(
                id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                kwargs={"env_cfg_entry_point": f"{__name__}:Go2FlatP0EnvCfgV{n}",
                        "rsl_rl_cfg_entry_point": f"{__name__}:Go2FlatP0PPORunnerCfgV{n}",
                        "default_agent": "rsl_rl", "sim2gate_variant": v})
    return None
