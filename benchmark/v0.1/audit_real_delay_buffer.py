"""Gate 5: real Isaac Lab DelayBuffer + RootStep DelayedDCMotor, CPU only.

Run inside Isaac Lab container using /isaac-sim/python.sh. No simulator launch,
GPU, training, or modifications to the installed sim2gate package.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from isaaclab.utils.buffers import DelayBuffer
from sim2gate.mvp.actuators import motor_class


class MinimalDCMotor:
    """Only the base DC motor is stubbed; delay logic and buffers are real."""
    def __init__(self, cfg):
        self.cfg = cfg
        self._num_envs = 64
        self._device = "cpu"
        self.computed_effort = None
        self.applied_effort = None

    def reset(self, env_ids):
        pass

    def compute(self, action, joint_pos, joint_vel):
        self.computed_effort = action.joint_efforts.clone()
        self.applied_effort = action.joint_efforts.clone()
        return action


def make_action(value):
    x = torch.full((64, 1), float(value), device="cpu")
    return SimpleNamespace(joint_positions=x.clone(), joint_velocities=x.clone(), joint_efforts=x.clone())


def run():
    torch.manual_seed(42)
    Motor = motor_class(MinimalDCMotor, DelayBuffer)
    cfg = dict(min_delay=0, max_delay=6, delay_ramp_start_steps=48000,
               delay_ramp_end_steps=76800, min_strength=1.0, max_strength=1.0)
    h0, d2 = Motor(SimpleNamespace(**cfg)), Motor(SimpleNamespace(**{**cfg, "max_delay": 0}))
    checks, observations = {}, []
    for steps, max_expected in [(0, 0), (48000, 0), (62400, 3), (76800, 6)]:
        h0.sim2gate_physics_steps = d2.sim2gate_physics_steps = steps
        h0.reset(None)
        d2.reset(None)
        checks[f"H0_config_max_{steps}"] = h0.current_max_delay() == max_expected
        checks[f"D2_config_zero_{steps}"] = d2.current_max_delay() == 0
        # Real buffer internal lag is intentionally not introspected: verify
        # observable command response instead.
        for motor, name in [(h0, "H0"), (d2, "D2")]:
            observed = []
            for i in range(12):
                a = make_action(i + 1)
                out = motor.compute(a, a.joint_positions, a.joint_velocities)
                observed.append(out.joint_positions[:, 0].clone())
                checks[f"{name}_targets_equal_{steps}_{i}"] = (
                    torch.equal(out.joint_positions, out.joint_velocities)
                    and torch.equal(out.joint_positions, out.joint_efforts))
            matrix = torch.stack(observed)
            # With delays sampled in [0,max_expected], an impulse-like unique
            # input at step t should emerge between t and t+max_expected.
            for t in range(7):
                expected = torch.arange(1, 13, dtype=matrix.dtype)
                value = matrix[t]
                checks[f"{name}_lag_bound_{steps}_{t}"] = bool(torch.all(value >= max(0, t + 1 - (max_expected if name == "H0" else 0)))) and bool(torch.all(value <= t + 1))
            if name == "D2":
                checks[f"D2_immediate_{steps}"] = bool(torch.allclose(matrix, torch.arange(1, 13, dtype=matrix.dtype).unsqueeze(1).expand_as(matrix)))
            observations.append({"physics_steps": steps, "recipe": name,
                                 "first_env_outputs": matrix[:8, 0].tolist()})
    # Ensure positive H0 delay after full ramp by looking for any lagged
    # output across 64 independently sampled environments.
    h0.sim2gate_physics_steps = 76800
    h0.reset(None)
    values = []
    for i in range(10):
        a = make_action(i + 1)
        values.append(h0.compute(a, a.joint_positions, a.joint_velocities).joint_positions[:, 0].clone())
    matrix = torch.stack(values)
    checks["H0_observed_nonzero_lag_after_ramp"] = bool(torch.any(matrix[8] < 9))
    result = {"gate": "Gate 5: real Isaac Lab DelayBuffer integration (CPU)",
              "status": "PASS" if all(checks.values()) else "FAIL",
              "checks_passed": sum(checks.values()), "checks_total": len(checks),
              "failed_checks": [k for k, v in checks.items() if not v],
              "observations": observations,
              "limits": ["Uses actual Isaac Lab DelayBuffer but minimal DC motor stub",
                         "Does not instantiate a robot or physics simulation",
                         "Does not test learned policy behavior or diagnostic accuracy"]}
    out = Path("benchmark/v0.1/GATE5_REAL_BUFFER_REPORT.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run())
