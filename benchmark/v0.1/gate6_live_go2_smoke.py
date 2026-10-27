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
            # Capture real DelayBuffer state on the articulated actuator.
            # The three channels must share a sampled lag for each environment.
            def capture_lags():
                return [buf.time_lags.detach().cpu().tolist() for buf in motor._delay_bufs]
            report["observations"][arm]["lags_before_explicit_reset"] = capture_lags()
            report["checks"][arm+"_lag_channel_alignment_initial"] = (
                all(lags == capture_lags()[0] for lags in capture_lags())
            )
            report["checks"][arm+"_buffer_capacity"] = all(
                buf.history_length == motor.cfg.max_delay for buf in motor._delay_bufs
            )

            # Sample reset lags via observed target behavior in Gate 5; here
            # verify actual actuator reset and physics stepping in a live scene.
            checkpoint(arm + "_before_reset")
            obs, info = env.reset()
            checkpoint(arm + "_reset_complete")
            report["checks"][arm+"_reset"] = obs is not None
            sampled_lags = capture_lags()
            report["observations"][arm]["lags_after_env_reset"] = sampled_lags
            report["checks"][arm+"_lag_channel_alignment_after_reset"] = all(
                channel == sampled_lags[0] for channel in sampled_lags
            )
            # Controlled counter probes on the real motor, without advancing physics.
            # These verify ramp sampling but do not replace command trace evidence.
            original_counter = motor.sim2gate_physics_steps
            ramp_probes = []
            try:
                for counter, expected_max in ((0, 0), (62400, 3), (76800, 6)):
                    motor.sim2gate_physics_steps = counter
                    motor.reset(None)
                    observed = capture_lags()
                    current_max = motor.current_max_delay()
                    ramp_probes.append({
                        "counter": counter,
                        "expected_max": expected_max,
                        "actual_max": current_max,
                        "sampled_lags": observed,
                    })
                    report["checks"][arm+"_ramp_"+str(counter)] = (
                        current_max == (expected_max if arm == "H0" else 0)
                        and all(ch == observed[0] for ch in observed)
                        and all(0 <= lag <= current_max for ch in observed for lag in ch)
                    )
            finally:
                motor.sim2gate_physics_steps = original_counter
                motor.reset(None)
            report["observations"][arm]["ramp_probes"] = ramp_probes
            checkpoint(arm + "_ramp_probes_complete")
            report["checks"][arm+"_sampled_lag_in_bounds"] = all(
                motor.cfg.min_delay <= lag <= motor.current_max_delay()
                for channel in sampled_lags for lag in channel
            )

            # Gate 6C: observational wrappers on this instance only. No changes
            # to production actuator code, and no optimizer/checkpoint writes.
            # Force lag=2 on H0 and lag=0 on D2 after reset to exercise the
            # buffer at a controlled test condition, independent of the ramp.
            forced_lag = 2 if arm == "H0" else 0
            for buf in motor._delay_bufs:
                buf.set_time_lag(forced_lag)
                buf.reset()
            report["observations"][arm]["trace_forced_lag"] = forced_lag
            trace = []
            trace_step = {"index": 0}
            channel_records = {}
            original_compute = motor.compute
            original_buffers = [b.compute for b in motor._delay_bufs]

            def sample(tensor):
                if tensor is None:
                    return None
                return tensor.detach().flatten()[:4].cpu().tolist()

            def trace_buffer(channel, original):
                def wrapped(data):
                    incoming = sample(data)
                    delayed = original(data)
                    channel_records[channel] = {
                        "input": incoming, "delayed": sample(delayed)
                    }
                    return delayed
                return wrapped

            for channel, buf in enumerate(motor._delay_bufs):
                buf.compute = trace_buffer(("position", "velocity", "effort")[channel],
                                           original_buffers[channel])

            def traced_motor_compute(control_action, joint_pos, joint_vel):
                channel_records.clear()
                output = original_compute(control_action, joint_pos, joint_vel)
                if len(trace) < 32:
                    trace.append({
                        "physics_call": trace_step["index"],
                        "lag": [int(b.time_lags[0].item()) for b in motor._delay_bufs],
                        "channels": dict(channel_records),
                        "computed_effort": sample(motor.computed_effort),
                        "applied_effort": sample(motor.applied_effort),
                        "output_effort": sample(output.joint_efforts),
                        "joint_position": sample(joint_pos),
                        "joint_velocity": sample(joint_vel),
                    })
                trace_step["index"] += 1
                return output

            motor.compute = traced_motor_compute
            action_dim = env.action_manager.total_action_dim
            for i in range(args.steps):
                # Small alternating actions create observable target changes.
                # This is a bounded test stimulus, not a trained policy.
                amplitude = 0.05 if i % 2 == 0 else -0.05
                actions = torch.full((args.num_envs, action_dim), amplitude, device=env.device)
                obs, reward, terminated, truncated, info = env.step(actions)
            report["checks"][arm+"_live_steps"] = obs is not None and torch.isfinite(reward).all().item()
            report["observations"][arm]["completed_steps"] = args.steps
            report["observations"][arm]["physics_step_counter"] = motor.sim2gate_physics_steps
            report["observations"][arm]["motor_trace"] = trace
            report["checks"][arm+"_trace_recorded"] = len(trace) > 0
            report["checks"][arm+"_trace_channels_present"] = (
                len(trace) > 0 and all(
                    set(item["channels"]) == {"position", "velocity", "effort"}
                    for item in trace
                )
            )
            report["checks"][arm+"_trace_lag"] = (
                len(trace) > 0 and all(
                    item["lag"] == [forced_lag] * 3 for item in trace
                )
            )
            # Behavioral proof requires a changed input and a corresponding
            # stale output for H0, while D2 remains immediate.
            def differs(left, right, tol=1e-6):
                return left is not None and right is not None and any(
                    abs(a - b) > tol for a, b in zip(left, right)
                )
            position_trace = [entry["channels"]["position"] for entry in trace]
            report["observations"][arm]["position_input_changes"] = sum(
                differs(position_trace[j]["input"], position_trace[j-1]["input"])
                for j in range(1, len(position_trace))
            )
            report["observations"][arm]["position_delay_mismatches"] = sum(
                differs(entry["input"], entry["delayed"]) for entry in position_trace
            )
            report["checks"][arm+"_varying_position_stimulus"] = (
                report["observations"][arm]["position_input_changes"] > 0
            )
            report["checks"][arm+"_delay_behavior"] = (
                report["observations"][arm]["position_delay_mismatches"] > 0
                if arm == "H0" else
                report["observations"][arm]["position_delay_mismatches"] == 0
            )
            # Gate 6D: two-environment selective reset, using the three real
            # delay buffers owned by the live articulated Go2 actuator.
            # Run after live stepping; synthetic inputs are confined to buffers.
            if args.num_envs == 2 and arm == "H0":
                isolation = []
                for channel, buf in enumerate(motor._delay_bufs):
                    buf.reset()
                    buf.set_time_lag(2)
                    width = int(motor._joint_vel.shape[1])  # actual live actuator joint tensor
                    for value in (10.0, 20.0, 30.0, 40.0):
                        buf.compute(torch.full((2, width), value, device=env.device))
                    lag_before = buf.time_lags.detach().cpu().tolist()
                    motor.reset(torch.tensor([0], device=env.device, dtype=torch.long))
                    # motor.reset resets all three buffers for env 0 only.
                    lag_after = buf.time_lags.detach().cpu().tolist()
                    output = buf.compute(torch.tensor(
                        [[99.0] * width, [50.0] * width], device=env.device
                    ))
                    env0 = float(output[0, 0].item())
                    env1 = float(output[1, 0].item())
                    isolation.append({
                        "channel": channel, "lag_before": lag_before,
                        "lag_after": lag_after, "env0_output": env0,
                        "env1_output": env1,
                        "pass": lag_before == [2, 2] and lag_after == [0, 2]
                                and abs(env0 - 99.0) < 1e-5
                                and abs(env1 - 30.0) < 1e-5,
                    })
                report["observations"][arm]["partial_reset_isolation"] = isolation
                report["checks"][arm+"_partial_reset_isolation"] = all(
                    item["pass"] for item in isolation
                )
                checkpoint(arm + "_partial_reset_checked")
            motor.compute = original_compute
            for buf, original in zip(motor._delay_bufs, original_buffers):
                buf.compute = original

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
