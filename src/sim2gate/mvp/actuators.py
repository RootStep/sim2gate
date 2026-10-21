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

Motor strength (P0 v5): with min_strength < max_strength, each reset also draws an output-torque scale per
environment, uniform between the ramped lower bound and max_strength (full strength before the ramp starts, the full
range after it ends), and the applied torque is multiplied by it, as check 1's motor-strength sweep does. The
computed (demanded) torque is multiplied too, so a weaker motor is not counted as torque-clipped by check 2. With the
defaults (1.0, 1.0) nothing is drawn and the motor behaves exactly as in P0 v3/v4.

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
        min_strength: float = 1.0
        """Lowest motor strength (output torque scale) drawn per episode (P0 v5; 1.0 = full strength always)."""
        max_strength: float = 1.0
        """Highest motor strength drawn per episode. The draw follows the same ramp as the delay."""

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
            self.sim2gate_strength = None              # per-env output scale, P0 v5 only

        def ramp_fraction(self):
            s0 = getattr(self.cfg, "delay_ramp_start_steps", 0) or 0
            s1 = getattr(self.cfg, "delay_ramp_end_steps", 0) or 0
            n = self.sim2gate_physics_steps
            if s1 > s0:
                return min(1.0, max(0.0, (n - s0) / (s1 - s0)))
            return 1.0 if n >= s0 else 0.0

        def current_max_delay(self):
            lo, hi = self.cfg.min_delay, self.cfg.max_delay
            return lo + int(round((hi - lo) * self.ramp_fraction()))

        def current_min_strength(self):
            """Weakest strength that may be drawn now: max_strength before the ramp, min_strength after it."""
            lo = getattr(self.cfg, "min_strength", 1.0)
            hi = getattr(self.cfg, "max_strength", 1.0)
            return hi - (hi - lo) * self.ramp_fraction()

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
            lo, hi = getattr(self.cfg, "min_strength", 1.0), getattr(self.cfg, "max_strength", 1.0)
            if lo < hi:                       # P0 v5: a strength per episode (no draw otherwise: v3/v4 unchanged)
                if self.sim2gate_strength is None:
                    self.sim2gate_strength = torch.full((self._num_envs, 1), float(hi), device=self._device)
                cur = self.current_min_strength()
                draw = cur + (hi - cur) * torch.rand(n, 1, device=self._device)
                if env_ids is None:
                    self.sim2gate_strength[:] = draw
                else:
                    self.sim2gate_strength[env_ids] = draw

        def compute(self, control_action, joint_pos, joint_vel):
            self.sim2gate_physics_steps += 1
            p, v, e = self._delay_bufs
            control_action.joint_positions = p.compute(control_action.joint_positions)
            control_action.joint_velocities = v.compute(control_action.joint_velocities)
            control_action.joint_efforts = e.compute(control_action.joint_efforts)
            out = super().compute(control_action, joint_pos, joint_vel)
            if self.sim2gate_strength is not None:    # as check 1's swept motor: scale the applied torque
                # the demand is scaled too, so check 2's "torque clipped" (demand != produced) means the DC motor's
                # limit was hit, not that the strength is below 1. Measurement only: the torque applied is unchanged
                # (fixed Oct 21, 2026, after the P0 v5b pilot's check 2 counted every step as clipped)
                self.computed_effort = self.computed_effort * self.sim2gate_strength
                self.applied_effort = self.applied_effort * self.sim2gate_strength
                out.joint_efforts = self.applied_effort
            return out

    return DelayedDCMotor


def with_delay(dc_cfg, min_delay, max_delay, ramp_steps=(0, 0), strength=(1.0, 1.0)):
    """A DelayedDCMotorCfg carrying every setting of an existing DCMotorCfg, plus the delay range (and ramp) and,
    for P0 v5, a motor-strength range drawn per episode on the same ramp."""
    _, Cfg = delayed_dc_motor()
    fields = {k: getattr(dc_cfg, k) for k in dc_cfg.__dataclass_fields__ if k != "class_type"}
    return Cfg(**fields, min_delay=min_delay, max_delay=max_delay, delay_ramp_start_steps=ramp_steps[0],
               delay_ramp_end_steps=ramp_steps[1], min_strength=float(strength[0]), max_strength=float(strength[1]))
