"""Gate 6: bounded live Go2 actuator integration, no policy training.

Run only in approved isolated Isaac Lab container. Requires a live Isaac Sim GPU.
Status is FAIL/BLOCKED until actual articulated robot checks succeed.
"""
import argparse
import datetime
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/rootstep-gate6-report.json")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if not (1 <= args.num_envs <= 2) or not (1 <= args.steps <= 32):
        parser.error("Hard limits: 1-2 environments, 1-32 steps")
    started = time.monotonic()
    report = {"gate": 6, "status": "BLOCKED", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "num_envs": args.num_envs, "steps_requested": args.steps, "checks": {}, "observations": {},
              "limitations": ["No PPO training", "No checkpoint writes", "Not a diagnostic benchmark performance test"]}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(stage):
        report["stage"] = stage
        report["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(tmp, path)
        print("[GATE6] " + stage, flush=True)

    def interrupted(signum, frame):
        report["status"] = "INTERRUPTED"
        report["signal"] = signum
        checkpoint("signal_received")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    checkpoint("script_started")
    app = None
    env = None
    try:
        checkpoint("before_app_launcher")
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False).app
        checkpoint("app_initialized")
        import torch
        from isaaclab.utils.buffers import DelayBuffer
        from isaaclab.actuators import DCMotor
        from sim2gate.mvp.actuators import delayed_dc_motor
        from sim2gate.v2 import tasks
        import gymnasium as gym
        tasks.register()
        torch.manual_seed(17)
        checkpoint("imports_and_seed_complete")
        Motor, _ = delayed_dc_motor()
        names = {"H0": "Go2FlatP0EnvCfgV5b", "D2": tasks._p2_names("Phi-delay", "nodelay")[0]}
        report["observations"]["torch"] = torch.__version__
        report["observations"]["source_commit"] = os.environ.get("ROOTSTEP_GATE6_COMMIT", "unknown")
        for arm, name in names.items():
            checkpoint(arm + "_before_config")
            cfg = getattr(tasks, name)()
            # Resolve all task presets, not only physics. The Arena task config
            # contains declarative physics and event placeholders that cannot be
            # passed directly to ManagerBasedRLEnv.
            from isaaclab_tasks.utils.hydra import resolve_presets
            from isaaclab.physics.physics_manager_cfg import PhysicsCfg
            from isaaclab.managers import EventTermCfg
            result = resolve_presets(cfg, selected=("isaacsim_physx",))
            cfg = cfg if result is None else result
            assert isinstance(cfg.sim.physics, PhysicsCfg)
            assert type(cfg.sim.physics).__name__ == "PhysxCfg"
            assert isinstance(cfg.events.base_com, EventTermCfg)
            assert cfg.sim.physics.gpu_max_rigid_patch_count == 327680
            physics_dict = cfg.sim.physics.to_dict()
            if arm == "H0":
                reference_physics = physics_dict
            else:
                assert physics_dict == reference_physics, "H0 and D2 physics configurations differ"
            report["observations"][arm + "_physics"] = {
                "resolver": "resolve_presets",
                "selected_preset": "isaacsim_physx",
                "selected_type": type(cfg.sim.physics).__name__,
                "gpu_max_rigid_patch_count": cfg.sim.physics.gpu_max_rigid_patch_count,
                "matches_H0": physics_dict == reference_physics,
            }
            checkpoint(arm + "_presets_resolved")
            cfg.scene.num_envs = args.num_envs
            cfg.seed = 17
            cfg.scene.env_spacing = 2.5
            # Task IDs are registered by tasks.register(); create via cfg's own class
            # rather than passing training runner or checkpoint paths.
            from isaaclab.envs import ManagerBasedRLEnv
            checkpoint(arm + "_before_env_creation")
            env = ManagerBasedRLEnv(cfg=cfg)
            checkpoint(arm + "_env_created")
            robot = env.scene["robot"]
            motor = robot.actuators["base_legs"]
            actual = type(motor)
            report["observations"][arm] = {"actuator_class": actual.__module__ + "." + actual.__qualname__,
                "config_class_identity": cfg.scene.robot.actuators["base_legs"].class_type is actual,
                "max_delay": motor.cfg.max_delay, "min_delay": motor.cfg.min_delay,
                "buffer_classes": [type(b).__module__ + "." + type(b).__qualname__ for b in motor._delay_bufs]}
            report["checks"][arm+"_actual_class"] = isinstance(motor, Motor) and issubclass(actual, DCMotor)
            report["checks"][arm+"_config_identity"] = cfg.scene.robot.actuators["base_legs"].class_type is actual
            report["checks"][arm+"_real_buffers"] = len(motor._delay_bufs) == 3 and all(isinstance(b, DelayBuffer) for b in motor._delay_bufs)
            report["checks"][arm+"_delay_range"] = (motor.cfg.min_delay, motor.cfg.max_delay) == ((0, 6) if arm == "H0" else (0, 0))
            report["checks"][arm+"_ramp"] = (motor.cfg.delay_ramp_start_steps, motor.cfg.delay_ramp_end_steps) == (48000, 76800)
            # Sample reset lags via observed target behavior in Gate 5; here
            # verify actual actuator reset and physics stepping in a live scene.
            checkpoint(arm + "_before_reset")
            obs, info = env.reset()
            checkpoint(arm + "_reset_complete")
            report["checks"][arm+"_reset"] = obs is not None
            action_dim = env.action_manager.total_action_dim
            for i in range(args.steps):
                actions = torch.zeros((args.num_envs, action_dim), device=env.device)
                obs, reward, terminated, truncated, info = env.step(actions)
            report["checks"][arm+"_live_steps"] = obs is not None and torch.isfinite(reward).all().item()
            report["observations"][arm]["completed_steps"] = args.steps
            report["observations"][arm]["physics_step_counter"] = motor.sim2gate_physics_steps
            checkpoint(arm + "_steps_complete")
            env.close()
            env = None
            checkpoint(arm + "_closed")
        # These live smoke checks do NOT yet instrument internal lag values,
        # three target traces, or partial reset isolation. Never call this full Gate 6 PASS.
        report["status"] = "PARTIAL" if all(report["checks"].values()) else "FAIL"
        report["missing_for_full_gate6"] = ["Live sampled-lag evidence at ramp milestones",
             "Input/output command and motor-effort traces", "Partial environment reset isolation"]
    except Exception as exc:
        report["status"] = "BLOCKED"
        report["error"] = type(exc).__name__ + ": " + str(exc)
        checkpoint("exception")
    finally:
        checkpoint("before_cleanup")
        if env is not None:
            try: env.close()
            except Exception: pass
        if app is not None:
            try: app.close()
            except Exception: pass
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        checkpoint("finished")
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PARTIAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
