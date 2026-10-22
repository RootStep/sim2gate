# Phase 1: the 20 clean robots (P0 v5b), Oct 21–22, 2026

**Result: complete. 20 robots qualified out of 22 trained.** Code `67205c4` (check 2 fix included), task
`Sim2Gate-Velocity-Flat-Go2-P0-v5b`, seeds 101–122, run under `PHASE1_PROTOCOL.md` section 4. Each robot was trained
with the full training record (EvidenceRecorder) and the monitor, then measured: nominal PhysX, check 1 on the v2
envelope, check 2, and nominal Newton, 256 episodes each where applicable.

| Seed | Nominal success (256 episodes) | Check 1 flags (v2 envelope) | Status |
|---|---|---|---|
| 101 | 0.957 | none | calibration set |
| 102 | 0.930 | none | calibration set |
| 103 | 0.969 | none | calibration set |
| 104 | 0.969 | none | calibration set |
| 105 | 0.953 | none | calibration set |
| 106 | 0.844 | not measured | **did not qualify** (< 0.85); replaced by 121 |
| 107 | 0.980 | none | calibration set |
| 108 | 0.941 | none | calibration set |
| 109 | 0.984 | none | calibration set |
| 110 | 0.934 | none | calibration set |
| 111 | 0.957 | none | calibration set |
| 112 | 0.832 | not measured | **did not qualify**; replaced by 122 |
| 113 | 0.914 | none | calibration set |
| 114 | 0.945 | none | calibration set |
| 115 | 0.980 | none | calibration set |
| 116 | 0.961 | none | calibration set |
| 117 | 0.957 | none | calibration set |
| 118 | 0.969 | none | calibration set |
| 119 | 0.980 | none | calibration set |
| 120 | 0.938 | none | calibration set |
| 121 | 0.969 | none | calibration set |
| 122 | 0.883 | **observation_noise_scale** | calibration set (flags are recorded, not used to reject) |

Success uses v1's frozen threshold (`frozen_success_p0v4.json`) until the v2 threshold is set from these robots.

- **Qualified:** 20 of 22. Success ranged from 0.883 to 0.984, with a median of 0.957.
- **Did not qualify:** 2 of 22 (0.844 and 0.832), both close to the bar. Replacement seeds 121 and 122 were used; 3 of
  5 spares remain.
- **Check 1 flags on clean robots:** 1 of 20, on sensor noise for seed 122. This is the first measured false-alarm
  count for check 1 on the v2 envelope. The pilot robots broke at 2.17× noise, just past the 2× edge, so noise is the
  condition where clean robots sit closest to the line.
- **Check 2 values and Newton results:** not read yet. They set Layer 1's limits, so the rules for using them are
  written and frozen first (`../../../LAYER1_LIMITS_PROTOCOL.md`, frozen Oct 22).
- **Check 2 fix:** confirmed on the GPU. Seed 101's torque-clipped share was 0.0025 (P0 v4 robots: about 0.002).

## Files (kept outside the repo)

| File | Size | SHA-256 |
|---|---|---|
| `phase1-clean.tgz` (records, logs, measurements, checkpoints; 1,704 files) | 2,597 MB | `36bb9b75caf2ab7d37482bf7dc79c5ddd48cacfd2bde3f3aac834ef5c23b712a` |
| `phase1-clean-curves.tgz` (training curves of all 22 robots and two P0 v4 references) | 52.9 MB | `3e50c16b1c950226f33cb23aaeb534a019b97da6d5c7f60880206389726cba65` |

The in-between checkpoints (41 per robot) stay on the instance under
`logs/rsl_rl/sim2gate_go2_flat_p0_v5b/`.
