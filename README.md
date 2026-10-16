# Sim2Gate

Training-aware diagnostics for sim-trained robot policies.

Sim2Gate looks at a reinforcement-learning policy trained in simulation, together with the
training setup that produced it, and reports why it is likely to fail on real hardware and what
to change in training. Every finding comes with the evidence that triggered it, a likely cause
and a concrete change to try.

> **Status: pre-alpha.** The training-signal logger below works; the hardware-readiness checks are not
> implemented yet, and nothing here has been validated against real hardware.

## What it will do

The first release targets quadruped locomotion policies trained with Isaac Lab and RSL-RL. It
reads evaluation results from [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) and
adds checks on the training recipe itself:

| Check | Looks for |
| --- | --- |
| Simulator exploits | Behavior that only works because of simulator artifacts: foot sliding, ground penetration, contact-force spikes, torques or velocities past actuator limits |
| Observation reliance | Dependence on signals that are exact in simulation but noisy or missing on hardware, such as ground-truth base velocity |

Reward attribution and training-history checks come later.

## What a report says

Each run ends in one of three findings, not a single score:

- **Risk found:** a specific failure risk, with evidence, likely cause and suggested change
- **Nothing found:** no failure detected within the tested conditions
- **Insufficient evidence:** the inputs can't support a call

A report informs a hardware decision. It does not replace a team's own safety checks.

## Training signals (experimental)

`sim2gate.training` logs signals from inside RSL-RL training, the part of the recipe that evaluation alone
can't see. For each PPO iteration it records, from the rollout PPO is about to train on:

- **ρ**, the correlation between the raw and the value-shaped advantage over the normalization group, and its
  regime (cancelled, partial, reversed). From the BSRS-in-PPO research behind Sim2Gate: ρ, computed before the
  update, predicted how much same-critic shaping changed PPO's gradient, optimizer step and policy.
- the zero-tail fraction (how much of the rollout saw no reward ahead of it)
- each reward term's share of the episode return, from Isaac Lab's `Episode_Reward/*` logs
- action std and learning rate

```python
from sim2gate.training import TrainingMonitor
monitor = TrainingMonitor(runner, "sim2gate_signals.jsonl")   # runner: rsl_rl OnPolicyRunner
runner.learn(1500)
```

Raw rewards are recorded as the environment hands them to the trainer, before RSL-RL adds its timeout bonus, so no
reward is reconstructed or rounded away (`raw_reward_source: "captured"` in each record).

The monitor is read-only: training with and without it gives bit-identical parameters (tested on CPU, full
training). On GPU, Isaac Lab training is not bit-reproducible run to run, so the GPU evidence is narrower: one PPO
update (`compute_returns` + `update`) replayed from the same saved state is bit-identical with and without Sim2Gate
(`--update-identity`); the monitor's per-step rollout hook and complete training are covered by the CPU test only. Its math
matches a reference implementation that passed an acceptance gate against rsl-rl-lib 5.0.1, 5.4.1 and 5.5.1, down
to the actor gradient. With Isaac Lab 3.x, use the launcher, which leaves Isaac Lab untouched:

```bash
pip install "sim2gate[rslrl]"
./isaaclab.sh -p -m sim2gate.training.isaaclab_launch --eta 0 -- \
    --task Isaac-Velocity-Flat-UnitreeGo2
```

`--eta` above 0 trains with same-critic BSRS (`sim2gate.training.BSRSPPO`), for research runs.
See [docs/EXPERIMENT_T_RUNBOOK.md](docs/EXPERIMENT_T_RUNBOOK.md). For a ready GPU workspace (Isaac Lab-Arena plus
Sim2Gate on NVIDIA Brev), see [deploy/brev](deploy/brev/README.md); `python -m sim2gate.training.smoke_go2` then runs
the Go2 smoke test with automatic pass/fail.

## Check 2: simulator exploits (experimental)

Measures, on a trained policy, the behaviors that only work because of simulator artifacts: foot slip (95th
percentile horizontal foot speed in contact), ground penetration, contact-force spikes (99th percentile foot force
in body weights), actuator limits (share of steps where the actuator model clips the torque demand or a joint
exceeds its speed limit; flagged above 1%), and action jitter (share of action power above a quarter of the
control frequency). In Isaac Lab 3.x, it runs Isaac Lab's own play entry point unchanged:

```bash
python -m sim2gate.checks.isaaclab_exploits --steps 1000 -- --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 64 \
    --checkpoint logs/rsl_rl/unitree_go2_flat/<run>/model_<iter>.pt
```

Name the policy explicitly (`--checkpoint`, or `--load_run`): without it Isaac Lab plays the newest run, and check 2
refuses to start unless you pass `--allow-latest`. The report records the loaded checkpoint's path and SHA-256 under
`checkpoint`, and the play arguments under `play_args`.

How to read the report:

- **When it measures.** Isaac Lab resets environments that ended inside `step()`, so reading the scene afterwards
  would show the respawned robot instead of the fall. Check 2 snapshots the scene after the reward computation and
  before any reset, on every step; `sources.state_read` in the report says which steps used which read.
- **Contact force** is the largest current sensor reading over every scene update inside the control step, in body
  weights computed from the masses in effect, randomized ones included. When Isaac Lab steps the physics itself
  (PhysX, or Newton with eager actuators) the scene updates after every physics substep, so every substep is seen.
  When Newton runs the whole decimation block in one call (all actuators CUDA-graph-safe), the scene updates once per
  control step and only the last substep is visible. `contact_force_substeps` in the report gives the samples per
  step, the covered share and which path ran, and a note appears when coverage is below 1. The sensor's history
  buffer is not used: what it spans depends on that update cadence. On Isaac Lab 3.0's Go2 flat task the default
  (Newton with Newton actuators) is the single-call path, so 1 of 4 substeps; `env.sim.use_newton_actuators=false`
  (same Newton physics, Isaac Lab's actuator path) or `physics=physx` gives 4 of 4. Full coverage was shown for those
  settings only, not for the default path. The report records the physics path and these overrides
  (`physics_context`).
- **Thresholds carry their measurement settings.** A thresholds file records the settings it was calibrated under
  (control step, contact threshold, foot radius, penetration tolerance, jitter window, flat ground, measurement
  version), the contact-force samples per step and the physics context (whether the engine ran the decimation
  block, and any physics overrides). All are required. Check 2 refuses a thresholds file whose settings differ from
  the run, and does not apply calibrated flags when the force sampling or the physics context differs.
- **Missing data is never "ok".** A channel that was not measured is `null`. The actuator flag is `ok` only when
  both channels (torque clipping and joint speed) were measured on every step; `coverage` gives the share for each.
  A violation in a partially measured run is still flagged.
- **Ground penetration** is measured only on flat ground (`terrain_type == "plane"`), from body-origin heights:
  foot origin minus the foot radius (`--foot-radius`, 0.022 m for the Go2), and other bodies' origins only. On any
  other terrain it is `null`.
- **Percentiles** use contact samples from the whole run; past 2 million samples they are thinned uniformly in
  time (`contact_sample_stride`).
- **Thresholds.** Only the actuator-limit flag has a fixed threshold (1% of steps). The others stay
  `uncalibrated` until you pass a versioned thresholds file. `sim2gate.checks.calibrate` makes one from the check 2
  reports of the calibration baselines (clean policies, P0 seeds 1 to 3), by the plan's rule: the largest value
  across the baselines times 1.5. It refuses fewer than three baselines, a repeated or unrecorded checkpoint, reports
  measured with different settings, and missing metrics, and it records each report's and checkpoint's SHA-256:

```bash
python -m sim2gate.checks.calibrate --out thresholds_go2_flat.json p0_s1.json p0_s2.json p0_s3.json
python -m sim2gate.checks.isaaclab_exploits --steps 1000 --thresholds thresholds_go2_flat.json -- \
    --task Isaac-Velocity-Flat-UnitreeGo2 --num_envs 64 --checkpoint <policy>.pt
```

These are indicators of simulator-dependent behavior, not proof of an exploit or of real-world transfer.

## Usage

```bash
pip install sim2gate
sim2gate preflight --help
```

The `preflight` command is a placeholder in this release.

## License

Apache 2.0. See [LICENSE](LICENSE).
