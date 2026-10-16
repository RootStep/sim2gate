"""A DC motor with a random command delay per episode, for training (plan change C6).

Isaac Lab's DelayedPDActuator delays an ideal PD motor; the Go2 task uses the DCMotor model (torque-speed limits).
DelayedDCMotor is that DCMotor with the same delay mechanism: the position, velocity and effort targets pass through
a delay buffer of whole physics steps, and each environment draws a new delay, uniformly in
[min_delay, max_delay], at every reset. Check 1's latency sweep (sim2gate.checks.isaaclab_conditions) delays the
same three targets the same way, with a fixed delay per environment.

Delay ramp (plan change C7): with delay_ramp_end_steps > 0, the largest delay drawn at a reset grows with the
motor's own count of physics steps: none before delay_ramp_start_steps, the full [min_delay, max_delay] from
delay_ramp_end_steps on, rising linearly in between. With both at 0 (the default) every reset draws from the full
range. P0 v4 uses the ramp in training only; its play_mode() switches it off, so evaluation always sees the full
range.

Built on first use, so this module imports without Isaac Lab.
"""
_BUILT = None


def delayed_dc_motor():
    """(DelayedDCMotor, DelayedDCMotorCfg)"""
    global _BUILT
    if _BUILT is not None:
        return _BUILT
    from isaaclab.actuators import DCMotor, DCMotorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.buffers import DelayBuffer
    DelayedDCMotor = motor_class(DCMotor, DelayBuffer)

    @configclass
    class DelayedDCMotorCfg(DCMotorCfg):
        class_type: type = DelayedDCMotor
        min_delay: int = 0
        """Minimum command delay, physics steps."""
        max_delay: int = 0
        """Maximum command delay, physics steps."""
        delay_ramp_start_steps: int = 0
        """Physics steps (this motor's compute calls) before any delay is drawn (C7 ramp; 0 = no ramp)."""
        delay_ramp_end_steps: int = 0
        """Physics steps from which the full delay range is drawn (C7 ramp; 0 = no ramp)."""

    _BUILT = (DelayedDCMotor, DelayedDCMotorCfg)
    return _BUILT


def motor_class(DCMotor, DelayBuffer):
    """DelayedDCMotor on the given DCMotor and DelayBuffer classes (Isaac Lab's, or fakes in tests)."""
    import torch

    class DelayedDCMotor(DCMotor):
        def __init__(self, cfg, *a, **k):
            super().__init__(cfg, *a, **k)
            self._delay_bufs = [DelayBuffer(cfg.max_delay, self._num_envs, device=self._device) for _ in range(3)]
            self.sim2gate_physics_steps = 0

        def current_max_delay(self):
            lo, hi = self.cfg.min_delay, self.cfg.max_delay
            s0 = getattr(self.cfg, "delay_ramp_start_steps", 0) or 0
            s1 = getattr(self.cfg, "delay_ramp_end_steps", 0) or 0
            n = self.sim2gate_physics_steps
            if s1 > s0:
                frac = min(1.0, max(0.0, (n - s0) / (s1 - s0)))
            else:
                frac = 1.0 if n >= s0 else 0.0
            return lo + int(round((hi - lo) * frac))

        def reset(self, env_ids):
            super().reset(env_ids)
            if env_ids is None or isinstance(env_ids, slice):
                env_ids, n = None, self._num_envs
            else:
                n = len(env_ids)
            lags = torch.randint(self.cfg.min_delay, self.current_max_delay() + 1, (n,), dtype=torch.int,
                                 device=self._device)
            for b in self._delay_bufs:
                b.set_time_lag(lags, env_ids)
                b.reset(env_ids)

        def compute(self, control_action, joint_pos, joint_vel):
            self.sim2gate_physics_steps += 1
            p, v, e = self._delay_bufs
            control_action.joint_positions = p.compute(control_action.joint_positions)
            control_action.joint_velocities = v.compute(control_action.joint_velocities)
            control_action.joint_efforts = e.compute(control_action.joint_efforts)
            return super().compute(control_action, joint_pos, joint_vel)

    return DelayedDCMotor


def with_delay(dc_cfg, min_delay, max_delay, ramp_steps=(0, 0)):
    """A DelayedDCMotorCfg carrying every setting of an existing DCMotorCfg, plus the delay range (and ramp)."""
    _, Cfg = delayed_dc_motor()
    fields = {k: getattr(dc_cfg, k) for k in dc_cfg.__dataclass_fields__ if k != "class_type"}
    return Cfg(**fields, min_delay=min_delay, max_delay=max_delay, delay_ramp_start_steps=ramp_steps[0],
               delay_ramp_end_steps=ramp_steps[1])
