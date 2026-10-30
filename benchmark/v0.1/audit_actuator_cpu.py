"""Gate 4: exercise actual DelayedDCMotor logic with CPU torch and fake motor/buffer.

No Isaac Sim, physics, GPU, policy training, or mutation of deployed recipes.
Run from repo root: PYTHONPATH=src python benchmark/v0.1/audit_actuator_cpu.py
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from sim2gate.mvp.actuators import motor_class


class FakeMotor:
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


class FakeDelayBuffer:
    def __init__(self, max_delay, num_envs, device):
        self.max_delay = max_delay
        self.num_envs = num_envs
        self.lags = torch.zeros(num_envs, dtype=torch.int)
        self.history = []

    def set_time_lag(self, lags, env_ids):
        if env_ids is None:
            self.lags[:] = lags
        else:
            self.lags[env_ids] = lags

    def reset(self, env_ids):
        self.history = []

    def compute(self, x):
        self.history.append(x.clone())
        result = torch.zeros_like(x)
        for i, lag in enumerate(self.lags.tolist()):
            index = len(self.history) - 1 - lag
            if index >= 0:
                result[i] = self.history[index][i]
        return result


def run():
    torch.manual_seed(17)
    Motor = motor_class(FakeMotor, FakeDelayBuffer)
    cfg = SimpleNamespace(min_delay=0, max_delay=6, delay_ramp_start_steps=48000,
                          delay_ramp_end_steps=76800, min_strength=1.0, max_strength=1.0)
    h0 = Motor(cfg)
    d2 = Motor(SimpleNamespace(**{**vars(cfg), "max_delay": 0}))
    checks = {}
    schedule = []
    for steps, expected in [(0, 0), (48000, 0), (62400, 3), (76800, 6)]:
        h0.sim2gate_physics_steps = steps
        d2.sim2gate_physics_steps = steps
        h0.reset(None)
        d2.reset(None)
        observed = h0._delay_bufs[0].lags
        d2_lags = d2._delay_bufs[0].lags
        checks[f"H0_bound_{steps}"] = h0.current_max_delay() == expected and bool(torch.all(observed <= expected))
        checks[f"D2_zero_{steps}"] = d2.current_max_delay() == 0 and bool(torch.all(d2_lags == 0))
        checks[f"shared_lags_{steps}"] = all(torch.equal(b.lags, observed) for b in h0._delay_bufs)
        schedule.append({"physics_steps": steps, "H0_max": expected,
                         "H0_sample_min": int(observed.min()), "H0_sample_max": int(observed.max()),
                         "D2_sample_max": int(d2_lags.max())})
    checks["H0_positive_lags_after_ramp"] = schedule[-1]["H0_sample_max"] > 0
    # Deterministic command propagation: force all buffers to 2-step delay.
    h0.reset(None)
    for b in h0._delay_bufs:
        b.set_time_lag(torch.full((64,), 2, dtype=torch.int), None)
    outputs = []
    for value in (1.0, 2.0, 3.0, 4.0):
        x = torch.full((64, 1), value)
        a = SimpleNamespace(joint_positions=x.clone(), joint_velocities=x.clone(), joint_efforts=x.clone())
        out = h0.compute(a, x, x)
        outputs.append(float(out.joint_positions[0, 0]))
        checks[f"three_targets_same_{int(value)}"] = bool(torch.equal(out.joint_positions, out.joint_velocities)) and bool(torch.equal(out.joint_positions, out.joint_efforts))
    checks["two_step_command_shift"] = outputs == [0.0, 0.0, 1.0, 2.0]
    result = {"gate": "CPU Gate 4: actuator implementation with fake buffers",
              "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "schedule": schedule, "two_step_outputs": outputs,
              "limitations": ["Fake buffer semantics; not Isaac Lab DelayBuffer integration",
                              "No simulation or policy training", "No GPU execution"]}
    path = Path("benchmark/v0.1/GATE4_ACTUATOR_CPU_REPORT.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run())
