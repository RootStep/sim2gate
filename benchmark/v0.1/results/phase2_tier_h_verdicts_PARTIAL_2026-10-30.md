# Tier H (historical) comparator table — PARTIAL, 2026-10-30

**Development only.** Recomputed on the CPU by `benchmark/v0.1/phase2_tier_h_verdicts.py` from the recorded Phase 2 reports, under the frozen v2 rules: success threshold 0.1187 m/s and Check 2 combined limit 1.0541. Nothing was tuned. For all 14 study robots, the recomputed nominal success and Check 1 flags equal the values the study driver recorded in `state.json`. **Check 2 for Φ-slip-mild and Φ-delay was measured under each arm's training physics** ([CHECK2_PHYSICS_CONFOUND_EVIDENCE.md](../CHECK2_PHYSICS_CONFOUND_EVIDENCE.md)). **PARTIAL:** the control report files were not in the transferred archive, so their rows are pending. Success-only flags are not computed until the 20-robot reference is assembled (spec section 6).

| policy | arm | nominal success | Check 1 | Check 2 (score / limit) | Check 2 measured under | Check 5 | any Sim2Gate flag |
|---|---|---|---|---|---|---|---|
| p0v5b_s101 | control | missing reports: nominal, sweep, check2, newton |  |  |  |  |  |
| p0v5b_s102 | control | missing reports: nominal, sweep, check2, newton |  |  |  |  |  |
| p0v5b_s103 | control | missing reports: nominal, sweep, check2, newton |  |  |  |  |  |
| p0v5b_s104 | control | missing reports: nominal, sweep, check2, newton |  |  |  |  |  |
| p0v5b_s105 | control | missing reports: nominal, sweep, check2, newton |  |  |  |  |  |
| s301 | R-slip | 1.000 (256 ep, 0 falls) | pass | FLAG (24.040 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s302 | R-slip | 1.000 (256 ep, 0 falls) | pass | FLAG (23.764 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s303 | R-slip | 0.992 (256 ep, 0 falls) | pass | FLAG (24.276 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s304 | R-slip | 0.980 (256 ep, 0 falls) | pass | FLAG (24.708 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s305 | R-slip | 1.000 (256 ep, 0 falls) | pass | FLAG (24.219 / 1.0541, foot_slip_p95_mps) | P0 v5b physics (reward change only) | pass | FLAG |
| s306 | Φ-slip-mild | 0.945 (256 ep, 0 falls) | pass | FLAG (2.053 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s307 | Φ-slip-mild | 0.957 (256 ep, 0 falls) | pass | FLAG (1.547 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s308 | Φ-slip-mild | 0.895 (256 ep, 0 falls) | pass | FLAG (1.609 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s309 | Φ-slip-mild | 0.938 (256 ep, 2 falls) | FLAG (observation_noise_scale) | FLAG (1.406 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s310 | Φ-slip-mild | 0.914 (256 ep, 0 falls) | pass | FLAG (3.221 / 1.0541, foot_slip_p95_mps) | TRAINING PHYSICS: static friction 0.3-0.6 | pass | FLAG |
| s311 | Φ-delay | 0.984 (256 ep, 0 falls) | FLAG (action_latency_ms) | FLAG (1.092 / 1.0541, action_jitter_high_freq_power_share) | TRAINING PHYSICS: no command delay | pass | FLAG |
| s312 | Φ-delay | 0.996 (256 ep, 0 falls) | FLAG (action_latency_ms) | FLAG (1.265 / 1.0541, action_jitter_high_freq_power_share) | TRAINING PHYSICS: no command delay | pass | FLAG |
| s313 | Φ-delay | missing: no final checkpoint (incomplete run) |  |  |  |  |  |
| s314 | Φ-delay | 0.992 (256 ep, 0 falls) | FLAG (action_latency_ms) | FLAG (1.893 / 1.0541, actuator_limit_step_share) | TRAINING PHYSICS: no command delay | pass | FLAG |
| s315 | Φ-delay | 1.000 (256 ep, 0 falls) | FLAG (action_latency_ms) | FLAG (1.346 / 1.0541, actuator_limit_step_share) | TRAINING PHYSICS: no command delay | pass | FLAG |
