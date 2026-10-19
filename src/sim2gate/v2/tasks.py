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

The v1 test set (sim2gate.mvp.tasks) is unchanged. Isaac Lab loads these through --external_callback
sim2gate.v2.tasks.register, which also registers the MVP tasks.
"""
P0_TASK_V5 = "Sim2Gate-Velocity-Flat-Go2-P0-v5"
P0_V5_STATIC_FRICTION = (0.35, 1.2)
P0_V5_DYNAMIC_FRICTION = (0.2625, 0.9)
P0_V5_PAYLOAD_KG = (0.0, 8.0)
P0_V5_MOTOR_STRENGTH = (0.8, 1.0)
P0_V5_ITERATIONS = 2000
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


_LAZY = {"Go2FlatP0EnvCfgV5": _p0v5_env_cfg, "Go2FlatP0PPORunnerCfgV5": _p0v5_agent_cfg}
_built = {}


def __getattr__(name):
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
