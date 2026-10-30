# Tier H (historical) comparator table: complete lock, 2026-10-30

**Development only; retrospective; post hoc.** The arms were selected after earlier pilots, and their outcomes had been seen before this analysis. This is not independent validation.

**How it was computed.** `benchmark/v0.1/phase2_tier_h_verdicts.py` recomputed every value on the CPU from the recorded Phase 2 reports. It used the frozen v2 rules: success threshold 0.1187 m/s, Check 2 combined limit 1.0541, and the production Check 1 and Check 5 scoring. Nothing was tuned or refit.

**What it reproduces.** For all 19 policies, the recomputed nominal success and Check 1 flags equal the study driver's `state.json` values. Every report names the policy's verified checkpoint.

**The success-only reference.** It is the 20 qualified P0 v5b robots (spec section 6): the 5 controls from their Phase 2 re-measure, and the other 15 from their Phase 1 nominal reports.
- Every Phase 1 report's per-robot tracking error equals the source value in `frozen_success_p0v5b.json`, and its checkpoint matches the register.
- A policy is flagged when its success is strictly below the reference minimum, 0.918 (p0v5b_s122).
- Sensitivity rules: the 5-control minimum (0.945), a leave-one-out reference, and the historical 0.85 cutoff.

**Comparator lock (spec G3).** This file fixes the tier-H comparators before any new trajectory. It supersedes `phase2_tier_h_verdicts_PARTIAL_2026-10-30.*`, which is kept as the earlier timestamped lock.

## By arm

Clopper–Pearson 95% intervals; descriptive for a small, selected cohort.

| Arm | n | Success-only flags (primary) | Sim2Gate any flag | Flags from checks not affected by the physics confound | Comparison status |
|---|---:|---|---|---|---|
| Controls | 5 | 0/5 (0.000–0.522) | 0/5 (0.000–0.522) | 0/5 | **In-sample** for both reference and Check 2 limit: descriptive only, not a false-alarm estimate |
| R-slip | 5 | 0/5 (0.000–0.522) | 5/5 (0.478–1.000) | 5/5: Check 2 slip (matched physics) | Fair: same evaluation physics as controls |
| Φ-slip-mild | 5 | 2/5 (0.053–0.853): s308, s310 | 5/5 (0.478–1.000) | 1/5: Check 1 noise (s309) | **Check 2 confounded:** measured on training friction 0.3–0.6 |
| Φ-delay | 4 | 0/4 (0.000–0.602) | 4/4 (0.398–1.000) | 4/4: Check 1 action latency | Check 1 fair (fixed nominal profile); Check 2 measured with delay off |
| s313 | — | missing | missing | — | No final checkpoint |

**Flawed policies missed by success-only but flagged by Sim2Gate:**
- **12/14 (0.572–0.982)** on any Sim2Gate check;
- **10/14 (0.419–0.916)** counting only checks unaffected by the physics confound: R-slip Check 2, Φ-delay Check 1, and Φ-slip-mild Check 1.

The 0.85 cutoff flags 0/14. The 5-control-minimum rule flags 3/14: s308, s309, s310.

**Interpretation limits:**
- Every flawed arm was kept because it stayed viable at nominal success. "Success misses it" is therefore a selected-cohort phenomenon, not a frequency.
- Intervention assignment is not behavioural truth. The independent symptom tests (tier R) are still to come.
- No false-alarm rate can be estimated from these controls.

## Per policy

| policy | arm | nominal success | success-only flag | Check 1 | Check 2 (score / limit) | Check 2 measured under | Check 5 | any Sim2Gate flag |
|---|---|---|---|---|---|---|---|---|
| p0v5b_s101 | control | 0.961 (256 ep, 1 falls) | pass | pass | pass (0.989 / 1.0541, action_jitter_high_freq_power_share) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
| p0v5b_s102 | control | 0.945 (256 ep, 0 falls) | pass | pass | pass (0.944 / 1.0541, foot_slip_p95_mps) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
| p0v5b_s103 | control | 0.969 (256 ep, 0 falls) | pass | pass | pass (0.915 / 1.0541, foot_slip_p95_mps) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
| p0v5b_s104 | control | 0.988 (256 ep, 0 falls) | pass | pass | pass (0.913 / 1.0541, foot_slip_p95_mps) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
| p0v5b_s105 | control | 0.961 (256 ep, 0 falls) | pass | pass | pass (0.905 / 1.0541, foot_slip_p95_mps) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
| s301 | R-slip | 1.000 (256 ep, 0 falls) | pass | pass | FLAG (24.040 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s302 | R-slip | 1.000 (256 ep, 0 falls) | pass | pass | FLAG (23.764 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s303 | R-slip | 0.992 (256 ep, 0 falls) | pass | pass | FLAG (24.276 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s304 | R-slip | 0.980 (256 ep, 0 falls) | pass | pass | FLAG (24.708 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s305 | R-slip | 1.000 (256 ep, 0 falls) | pass | pass | FLAG (24.219 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s306 | Φ-slip-mild | 0.945 (256 ep, 0 falls) | pass | pass | FLAG (2.053 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s307 | Φ-slip-mild | 0.957 (256 ep, 0 falls) | pass | pass | FLAG (1.547 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s308 | Φ-slip-mild | 0.895 (256 ep, 0 falls) | FLAG | pass | FLAG (1.609 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s309 | Φ-slip-mild | 0.938 (256 ep, 2 falls) | pass (min5: FLAG) | FLAG (observation_noise_scale) | FLAG (1.406 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s310 | Φ-slip-mild | 0.914 (256 ep, 0 falls) | FLAG | pass | FLAG (3.221 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s311 | Φ-delay | 0.984 (256 ep, 0 falls) | pass | FLAG (action_latency_ms) | FLAG (1.092 / 1.0541, action_jitter_high_freq_power_share) | TRAINING PHYSICS: no command delay | pass | FLAG |
| s312 | Φ-delay | 0.996 (256 ep, 0 falls) | pass | FLAG (action_latency_ms) | FLAG (1.265 / 1.0541, action_jitter_high_freq_power_share) | TRAINING PHYSICS: no command delay | pass | FLAG |
| s313 | Φ-delay | missing: no final checkpoint (incomplete run) |  |  |  |  |  |  |
| s314 | Φ-delay | 0.992 (256 ep, 0 falls) | pass | FLAG (action_latency_ms) | FLAG (1.893 / 1.0541, actuator_limit_step_share) | TRAINING PHYSICS: no command delay | pass | FLAG |
| s315 | Φ-delay | 1.000 (256 ep, 0 falls) | pass | FLAG (action_latency_ms) | FLAG (1.346 / 1.0541, actuator_limit_step_share) | TRAINING PHYSICS: no command delay | pass | FLAG |
