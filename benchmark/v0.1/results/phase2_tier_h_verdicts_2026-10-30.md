# Tier H (historical) comparator table — Sim2Gate side complete, 2026-10-30

**Development only; retrospective; post hoc.** Recomputed on the CPU by `benchmark/v0.1/phase2_tier_h_verdicts.py` from the recorded Phase 2 reports, under the frozen v2 rules: success threshold 0.1187 m/s, Check 2 combined limit 1.0541, and the production Check 1 and Check 5 scoring. Nothing was tuned or refit. For all 19 policies, the recomputed nominal success and Check 1 flags equal the study driver's `state.json` values. Every report names the policy's verified checkpoint.

- **Check 2 physics:** Φ-slip-mild and Φ-delay were measured under each arm's training physics ([CHECK2_PHYSICS_CONFOUND_EVIDENCE.md](../CHECK2_PHYSICS_CONFOUND_EVIDENCE.md)). Their Check 2 rows are a confounded historical comparator.
- **Controls are in-sample:** the five controls are among the 20 robots that set the Check 2 limit. Their passes are descriptive, not a false-alarm estimate.
- **Success-only flags are not yet computed.** They need the 20-robot healthy reference (spec section 6), which is waiting on the Phase 1 nominal reports.
- **Supersedes** `phase2_tier_h_verdicts_PARTIAL_2026-10-30.*` (committed earlier, 14 study robots only). That file is kept as the earlier timestamped lock.

| policy | arm | nominal success | Check 1 | Check 2 (score / limit) | Check 2 measured under | Check 5 | any Sim2Gate flag |
|---|---|---|---|---|---|---|---|
| p0v5b_s101 | control | 0.961 (256 ep, 1 falls) | pass | pass (0.989 / 1.0541, action_jitter_high_freq_power_share) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
| p0v5b_s102 | control | 0.945 (256 ep, 0 falls) | pass | pass (0.944 / 1.0541, foot_slip_p95_mps) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
| p0v5b_s103 | control | 0.969 (256 ep, 0 falls) | pass | pass (0.915 / 1.0541, foot_slip_p95_mps) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
| p0v5b_s104 | control | 0.988 (256 ep, 0 falls) | pass | pass (0.913 / 1.0541, foot_slip_p95_mps) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
| p0v5b_s105 | control | 0.961 (256 ep, 0 falls) | pass | pass (0.905 / 1.0541, foot_slip_p95_mps) | P0 v5b: static friction 0.35-1.2, delay 0-30 ms | pass | pass |
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
