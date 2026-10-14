"""Registered, versioned task variants for the MVP test set (docs/mvp-plan, change C4).

    P0 v1 = Isaac Lab's Go2 flat velocity task (Isaac-Velocity-Flat-UnitreeGo2) + Isaac Lab's own foot-slip penalty
            (mdp.feet_slide on the four feet, weight P0_FEET_SLIDE_WEIGHT)

The stock task has no foot-slip penalty, and the plan defines P3 as "foot-slip penalty removed" (plus low ground
friction), so the clean baseline must have one. The weight -0.25 is a proposed starting value, taken from Isaac
Lab's H1 configuration; it is not a validated Go2 setting. Whether P0 v1 is an acceptable clean baseline is decided
by the criteria in sim2gate.mvp.baselines.BASELINE_CRITERIA, fixed before training.

Isaac Lab loads these through its --external_callback hook (train and play):

    --external_callback sim2gate.mvp.tasks.register

Nothing is imported from Isaac Lab until the task is built, so this module imports without it. Changing a
variant means a new version (new task id); a registered version never changes.
"""

P0_TASK = "Sim2Gate-Velocity-Flat-Go2-P0-v1"
P0_FEET_SLIDE_WEIGHT = -0.25

VARIANTS = {
    P0_TASK: {
        "policy": "P0", "version": 1, "base_task": "Isaac-Velocity-Flat-UnitreeGo2",
        "changes": [{"what": "add reward term rewards.feet_slide", "func": "isaaclab_tasks.core.velocity.mdp.feet_slide",
                     "weight": P0_FEET_SLIDE_WEIGHT, "feet": ".*_foot", "sensor": "contact_forces",
                     "weight_status": "proposed starting value (Isaac Lab H1), not validated for the Go2"}],
        "experiment_name": "sim2gate_go2_flat_p0_v1",
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


def _p0_agent_cfg():
    from isaaclab.utils import configclass
    from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg

    @configclass
    class Go2FlatP0PPORunnerCfgV1(UnitreeGo2FlatPPORunnerCfg):
        def __post_init__(self):
            super().__post_init__()
            self.experiment_name = VARIANTS[P0_TASK]["experiment_name"]

    return Go2FlatP0PPORunnerCfgV1


_LAZY = {"Go2FlatP0EnvCfgV1": _p0_env_cfg, "Go2FlatP0PPORunnerCfgV1": _p0_agent_cfg}
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
    if P0_TASK not in gym.registry:
        gym.register(
            id=P0_TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{__name__}:Go2FlatP0EnvCfgV1",
                    "rsl_rl_cfg_entry_point": f"{__name__}:Go2FlatP0PPORunnerCfgV1",
                    "default_agent": "rsl_rl", "sim2gate_variant": VARIANTS[P0_TASK]})
    return None
