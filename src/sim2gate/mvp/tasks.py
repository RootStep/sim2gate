"""Registered, versioned task variants for the MVP test set (docs/mvp-plan, change C4).

    P0 v1 = Isaac Lab's Go2 flat velocity task (Isaac-Velocity-Flat-UnitreeGo2) + Isaac Lab's own foot-slip penalty
            (mdp.feet_slide on the four feet, weight P0_FEET_SLIDE_WEIGHT), the task's 300 training iterations
            (change C4; superseded by v2: seed 2 fell in 15 of 64 episodes)
    P0 v2 = P0 v1 trained for 1000 iterations instead of 300; nothing else changes (change C5)

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
P0_TASK = P0_TASK_V2                     # the current P0
P0_FEET_SLIDE_WEIGHT = -0.25
P0_V2_ITERATIONS = 1000

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
        "max_iterations": P0_V2_ITERATIONS, "experiment_name": "sim2gate_go2_flat_p0_v2", "status": "current",
    },
}


def _p0_env_cfg():
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.core.velocity import mdp
    from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

    @configclass
    class Go2FlatP0EnvCfgV1(UnitreeGo2FlatEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.rewards.feet_slide = RewardTermCfg(
                func=mdp.feet_slide, weight=P0_FEET_SLIDE_WEIGHT,
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                        "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")})

    return Go2FlatP0EnvCfgV1


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


# P0 v1 and v2 share the environment (the same penalty); they differ only in the training budget
_LAZY = {"Go2FlatP0EnvCfgV1": _p0_env_cfg, "Go2FlatP0EnvCfgV2": _p0_env_cfg,
         "Go2FlatP0PPORunnerCfgV1": lambda: _p0_agent_cfg(P0_TASK_V1),
         "Go2FlatP0PPORunnerCfgV2": lambda: _p0_agent_cfg(P0_TASK_V2)}
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
