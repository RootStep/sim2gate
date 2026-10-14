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

- Two runs per policy, 64 episodes each, nominal profile, same command set and seed: `physics=isaacsim_physx` and
  `physics=newton_mjwarp`. Newton keeps Isaac Lab's own solver settings for the task (recorded with its versions).
- Both runs use Isaac Lab's DC motor model (`env.sim.use_newton_actuators=false`, the same for every policy). The
  Go2 task otherwise runs its motors natively in Newton, which would change the motor code along with the engine.
- Parity is checked from the two raw files: identical task, checkpoint, command set, control and physics step,
  decimation, seed, environment count, profile and environment config (hashed without the physics settings); no
  physics override other than the engine and the fixed motor setting; no varied conditions.

## Open: P1 and P2 against this P0

At the pinned Isaac Lab commit the Go2 flat task trains with a fixed friction (0.8, no randomization) and no action
delay. The plan defines P1 as "friction and added-mass randomization set to a near-zero range" and P2 as "action
delay fixed at 0 in training", which presume a default that randomizes both. Against the current P0, P1 differs only
in mass, and P2 is identical to P0. Running check 1 on P0 v2 seeds 1–3 shows what that means in practice: P0 v2's
friction and latency cliffs are, in effect, P1's friction cliff and P2's latency cliff. A decision is needed before
the code freeze.
