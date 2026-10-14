# Check 2 calibration: P0 v2, seeds 1–3 (Oct 14, 2026)

The check 2 thresholds for the MVP, set from the calibration baselines under plan changes C3–C5. These files are
copied unmodified from the run bundle `s2g-p0v2-2026-10-14.tgz` (SHA-256
`87c5551888da632f3380d1895b55bb900442b02a283007cd9faff798ffe37676`, 100 files, all matching its SHA256SUMS), which
also holds the checkpoints, run folders and full logs.

- **Task:** `Sim2Gate-Velocity-Flat-Go2-P0-v2` (feet_slide −0.25, 1,000 iterations); every run folder's saved
  `agent.yaml` shows `max_iterations: 1000`, and every `env.yaml` shows `feet_slide` with weight −0.25.
- **Code:** Sim2Gate `e7716df`, Isaac Lab `91a0906`, Isaac Lab-Arena `2cc9e97`, rsl-rl-lib 5.4.1.
- **Physics:** Isaac Sim PhysX (`physics=isaacsim_physx`); check 2 saw all 4 physics updates of every control step.
- **Measurement:** check 2 measurement version 3, foot geometry from the authored collision shapes, fixed command set
  of 64 commands (seed 20261012, SHA-256 `56222a90…`), 1,000 control steps.

| Seed | Checkpoint | Tracking error | Falls | Sustained slip p95 | Accepted |
|---|---|---|---|---|---|
| 1 | `2c827cd6…` (model_999) | 0.062 m/s | 0 | 0.092 m/s | yes |
| 2 | `0691617d…` (model_999) | 0.057 m/s | 0 | 0.090 m/s | yes |
| 3 | `3d4f90bb…` (model_999) | 0.064 m/s | 0 | 0.093 m/s | yes |

Seed 3 finished all 1,000 iterations, then Isaac Sim crashed in `SimulationApp.close()` (exit 1); the driver kept
the final checkpoint and recorded `train_shutdown_crash` (C5).

| Metric | s1 | s2 | s3 | Threshold (max × 1.5) |
|---|---|---|---|---|
| foot_slip_p95_mps | 0.108 | 0.109 | 0.158 | 0.237 |
| ground_penetration_m | 0.0021 | 0.0018 | 0.0023 | 0.0035 |
| contact_force_p99_bw | 0.909 | 0.935 | 1.018 | 1.527 |
| action_jitter_high_freq_power_share | 0.0143 | 0.0138 | 0.0167 | 0.0251 |
| actuator_limit_step_share | 0.052 | 0.026 | 0.041 | 0.078 |

Verified on receipt: each checkpoint's hash matches the manifest and the hash its check 2 report measured;
recalibrating from the three reports reproduces `thresholds.json` exactly; `load_thresholds` accepts it; no foot was
counted in contact while more than 1 cm above the ground.

Not frozen yet: the checks and thresholds are frozen together in plan step 3. If check 2's measurement changes
before then, the thresholds are recalculated from these same three checkpoints, without retraining.
