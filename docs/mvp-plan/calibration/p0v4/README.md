# Calibration: P0 v4, seeds 1–3 (Oct 15, 2026)

The check 2 thresholds and the episode-success reference for checks 1 and 5, set from the calibration baselines
under plan changes C3–C8. They replace the P0 v2 calibration (`../p0v2`), which C6 superseded. The files here are
copied unmodified from the run bundle `s2g-p0v4-2026-10-15.tgz` (SHA-256
`0a9d55000b6f4ed7fff411aaf68567cd3a07fe7473622f116891512a741335c9`, 189 files, all matching its SHA256SUMS), which
also holds the checkpoints, run folders and full logs. `checks15/` holds the checks 1 and 5 files, including the
raw episode records they were scored from.

- **Task:** `Sim2Gate-Velocity-Flat-Go2-P0-v4` (C7): feet_slide −0.25; friction 0.4–1.2 (dynamic 0.3–0.9) from
  the first round; a 0–30 ms command delay (0–6 physics steps), off before round 500 and ramped in to round 800;
  2,000 rounds. Every run folder's saved `env.yaml` shows the `DelayedDCMotor` with `max_delay: 6` and the ramp at
  48,000–76,800 physics steps, and `use_newton_actuators: false`; every `agent.yaml` shows `max_iterations: 2000`.
- **Code:** training and all measurements at Sim2Gate `86f2c7d`; the checks 1 and 5 scores and the success
  reference at `4cf7ecf` (C8, `checks15 --score-only`, no new measurement). Isaac Lab `91a0906`, Isaac Lab-Arena
  `2cc9e97`, rsl-rl-lib 5.4.1. (The bundle's `sim2gate-versions.txt` was written by the `4cf7ecf` update.)
- **Physics:** check 2 and check 1 on Isaac Sim PhysX; check 5 on PhysX and Newton / MuJoCo-Warp, with the
  PhysX joint and body ordering on both.
- **Commands:** the fixed set of 64 (seed 20261012, SHA-256 `56222a90…`); 1,000 control steps (one 20 s episode).

## Baselines (C4 criteria, measured by check 2)

| Seed | Checkpoint | Tracking error | Falls | Sustained slip p95 | Accepted |
|---|---|---|---|---|---|
| 1 | `db9a4755…` (model_1999) | 0.071 m/s | 0 | 0.087 m/s | yes |
| 2 | `8a3f3615…` (model_1999) | 0.066 m/s | 0 | 0.089 m/s | yes |
| 3 | `1e721243…` (model_1999) | 0.081 m/s | 0 | 0.085 m/s | yes |

Seed 2 finished all 2,000 rounds, then Isaac Sim crashed in `SimulationApp.close()` (exit 1); the driver kept the
final checkpoint and recorded `train_shutdown_crash` (C5). Training tracking error (with noise and randomization
on) was 0.11–0.12 m/s at round 500, rose to 0.13–0.16 m/s while the delay ramped in, and ended at 0.12–0.14 m/s:
the robots walked before the delay arrived, as C7 intended.

## Check 2 thresholds (max over the three baselines × 1.5)

| Metric | s1 | s2 | s3 | Threshold | P0 v2 threshold |
|---|---|---|---|---|---|
| foot_slip_p95_mps | 0.103 | 0.105 | 0.107 | 0.161 | 0.237 |
| ground_penetration_m | 0.0040 | 0.0027 | 0.0027 | 0.0059 | 0.0035 |
| contact_force_p99_bw | 1.017 | 1.027 | 1.145 | 1.718 | 1.527 |
| action_jitter_high_freq_power_share | 0.0106 | 0.0108 | 0.0102 | 0.0162 | 0.0251 |
| actuator_limit_step_share | 0.0020 | 0.0023 | 0.0022 | 0.0034 | 0.0779 |

## Episode-success reference (`checks15/success.json`)

Episode success: no fall and mean tracking error at most 1.5 × 0.0755 = **0.1133 m/s**. The reference is the mean
over the 191 nominal sweep episodes without a fall (64, 63 and 64). Seed 2 fell in one nominal episode (env 26,
after 130 of 1,000 steps, mean error 0.311 m/s), within C4's tolerance of one; under C8 it is listed and left out.

## Checks 1 and 5: no seed flagged

| Seed | Check 1 nominal success | Check 1 cliffs (none inside the realistic range) | Check 5 Newton fall-free | Check 5 error ratio |
|---|---|---|---|---|
| 1 | 0.984 | friction 0.267, motor 0.80, latency 45 ms | 1.000 | 1.585 |
| 2 | 0.984 | friction 0.267 | 0.984 | 1.466 |
| 3 | 0.953 | friction 0.333, mass 3.5 kg, motor 0.825, latency 40 ms, noise 2.33× | 1.000 | 1.488 |

Check 5's parity passed on every seed. See `observations.md` for what to watch in the blinded run.

## Verified on receipt

- Every file matches the bundle's SHA256SUMS; each checkpoint's hash matches the baselines manifest, the checks 1
  and 5 manifest, and the hash each raw file and check 2 report measured.
- Recalibrating from the three check 2 reports reproduces `thresholds.json` exactly (only paths and the creation
  time differ); `load_thresholds` accepts it; no foot was counted in contact while more than 1 cm above the ground.
- Rebuilding `success.json` from the three sweep files reproduces it exactly, and rescoring every check 1 and check
  5 file from its raw files reproduces it exactly.

Not frozen yet: the checks and thresholds are frozen together in plan step 3. If a measurement changes before then,
the thresholds are recalculated from these same three checkpoints, without retraining.
