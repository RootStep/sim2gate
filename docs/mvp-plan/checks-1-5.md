# Checks 1 and 5: how the plan is implemented

Code: `sim2gate/checks/robustness.py` (check 1 grid and scoring), `sim2gate/checks/transfer.py` (check 5 parity and
scoring), `sim2gate/checks/isaaclab_conditions.py` (the Isaac Lab measurement), `sim2gate/mvp/checks15.py` (the driver).
Where the frozen plan leaves a choice open, the reading taken is listed here. None of them changes a threshold,
flag rule or acceptance criterion; they are for VJ to confirm at the final review.

## Shared: the nominal profile and episode success

Both checks run the policy under one **nominal profile** (`sim2gate-nominal-v1`), applied to the environment config
before the environment is built:

| Setting | Nominal profile | Why |
|---|---|---|
| Base mass, centre of mass | the model's values (the task's startup randomization removed) | "all others at nominal"; and identical across engines for check 5 |
| Robot contact material | static 0.8 / dynamic 0.6 (the task's training value); ground 1.0, multiply | the effective friction is the robot's value |
| Observation noise | on, at the task's own level | the plan's noise sweep starts from 1x the task's default noise |
| Pushes | off | as in Isaac Lab's play mode; not part of any check |
| Motors | Isaac Lab's DC motor model, plus a per-environment delay and output scale at nominal (none, 1.0) | the same motor code under every condition and both engines |
| Reset randomization (pose, velocity, joints) | kept | part of the episode |

This differs from check 2, which runs Isaac Lab's play defaults (startup mass randomization kept, noise off). Check
2's thresholds were calibrated under those defaults, so the two are kept separate.

**Episode success** (plan: no fall in the 20 s episode, and mean tracking error at most 1.5 x the clean baseline's
mean error under nominal conditions). The clean baseline's error is measured by check 1's own nominal group on
P0 seeds 1–3 (`success.json`, set once, never overwritten), so success is judged under the conditions it was
calibrated in. For checks 1 and 5 a condition "meets nominal success" when at least 80% of its episodes succeed.

## Check 1

- **Grid.** Each parameter goes from nominal towards harder conditions, 10 values, the last at nominal + 1.5 x
  (plausible edge − nominal); the plausible edge is then exactly the 7th value. Friction goes down from the task's
  0.8 to 0.2 (the plausible range's upper end, 1.2, makes walking easier); mass 0 to 4.5 kg; motor strength 1.0 to
  0.775; latency 0 to 45 ms in whole 5 ms physics steps; noise 1x to 2.5x.
- **One run per policy.** All 46 groups (one shared nominal group plus 9 values for each of 5 parameters) run
  together, 64 episodes each, 2,944 environments; group *g* holds environments 64*g* to 64*g*+63 and environment *i*
  gets command *i* mod 64 from the fixed set.
- **How each parameter is applied.** Friction and added mass through Isaac Lab's own event terms
  (`randomize_rigid_body_material`, `randomize_rigid_body_mass` with one value per group); motor strength scales
  the DC motor's output torque after its torque-speed limits (equivalent to scaling gains and limits); latency
  delays the motor's position, velocity and effort targets by whole physics steps; noise scales each policy noise
  term's range per environment. The applied values are read back into the raw file.
- **Cliff and flag** as in the plan. If nominal success is below 80%, check 1 is invalid ("policy not ready", exit
  code 3), not scored.
- **Not yet built:** the report-only joint sweep of all five parameters.

## Check 5

- Flagged if fewer than 80% of Newton episodes end without a fall, or the Newton/PhysX tracking-error ratio
  exceeds 2 (C6). The PhysX run must reach 80% episode success first ("policy not ready" otherwise).
- Two runs per policy, 64 episodes each, nominal profile, same command set and seed: `physics=isaacsim_physx` and
  `physics=newton_mjwarp`. Newton keeps Isaac Lab's own solver settings for the task (recorded with its versions).
- Both runs use Isaac Lab's DC motor model (`env.sim.use_newton_actuators=false`, the same for every policy). The
  Go2 task otherwise runs its motors natively in Newton, which would change the motor code along with the engine.
- **Same joint order on both engines.** PhysX lists the Go2's joints by type (all hips, then thighs, then calves);
  MuJoCo-Warp lists them leg by leg. Without correction a PhysX-trained policy's actions reach the wrong joints on
  Newton and the robot falls at once: on Oct 14 every P0 v2 seed had 0% success on Newton, and a Newton-trained
  policy failed the same way on PhysX. Both runs therefore use the training engine's order
  (`env.scene.robot.joint_ordering=physx`, `env.scene.robot.body_ordering=physx`), as Isaac Lab 3.0's guide
  "Transfer Policies Between PhysX and Newton" prescribes, and parity requires identical joint and body names in
  both raw files. `sim2gate.checks.robot_params` reads the robot back from each engine and lists any difference.
- Parity is checked from the two raw files: identical task, checkpoint, command set, control and physics step,
  decimation, seed, environment count, profile and environment config (hashed without the physics settings); no
  physics override other than the engine and the fixed motor setting; no varied conditions.

## Resolved by C6

- **P1 and P2.** P0 v2 never randomized friction and had no action delay, so P2 was identical to P0, and check 1
  flagged every P0 v2 seed on latency (cliffs 20–25 ms). P0 v3 trains with friction 0.4–1.2 and a 0–30 ms delay;
  P1 and P2 remove them.
- **Check 5's first rule** counts Newton falls (fall-free rate at least 80%); tracking is judged by the error-ratio
  rule alone. See CHANGES.md, C6.
