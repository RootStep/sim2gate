# Phase 2 arm pilots (part A), Oct 23–24, 2026

**Result: 1 of 5 planted interventions passed (R-slip at 10%). The other 4 were dropped under the frozen rules.**
Part A (`PHASE2_PART_A.md`, SHA-256 `d403db02…`) said that if two or more flawed arms are dropped, C2 and C3 are
reported as **not tested**. That rule now applies to phase 2 as designed. These are development robots: excluded from
all scoring, and never study robots or test robots.

Code `745872b`; smoke test passed Oct 23 (every arm's saved config matched its candidate; 11 identical reward terms
everywhere). The run was stopped overnight and resumed on Oct 24; seed 216 was retrained from the start. The driver
reports every robot, including failures.

## Hop reference

The 5 controls (101–105) were re-measured on check 2. The maximum of each policy-level statistic:

| Measure | Controls' maximum | Hop threshold |
|---|---|---|
| Flight share at low speed | 0.0020 | 0.02 (the floor applied) |
| Vertical-speed RMS at low speed | 0.0603 m/s | 0.0904 m/s |

## Results

Qualify means at least 85% success over 256 episodes under the v2 threshold. Symptom thresholds are from part A:
slip above 0.1185 m/s, jitter above 0.00986, hop as above.

| Arm | Candidate | Seed | Success | Qualifies | Intended symptom | Candidate result |
|---|---|---|---|---|---|---|
| R-slip | `feet_slide` × 0.1 | 201 | 0.996 | yes | yes | **passed: setting k0.1** |
| | | 202 | 0.996 | yes | yes | |
| Φ-slip | static friction 0.2–0.5 | 205 | 0.828 | no | yes | failed |
| | | 206 | 0.000 | no | yes | |
| | static friction 1.0–1.2 | 207 | 0.973 | yes | no | failed: **arm dropped** |
| | | 208 | 1.000 | yes | no | |
| R-jitter | `action_rate_l2` × 0.1 | 209 | 0.977 | yes | yes | failed |
| | | 210 | 0.984 | yes | no | |
| | `action_rate_l2` × 0.03 | 211 | 0.984 | yes | no | failed: **arm dropped** |
| | | 212 | 0.992 | yes | no | |
| Φ-jitter | no command delay | 213 | 0.992 | yes | no | failed: **arm dropped** |
| | | 214 | 1.000 | yes | yes | |
| R-hop | `feet_air_time` × 10 | 215 | 0.910 | yes | no | failed |
| | | 216 | 0.984 | yes | no | |
| | `feet_air_time` × 30 | 217 | 0.586 | no | no | failed: **arm dropped** |
| | | 218 | 0.016 | no | no | |

## Observations, reported only

- **R-slip.** At 10% slip penalty both robots slid beyond every clean robot, yet tracked speed better than any clean
  robot (99.6% success; clean robots 88–98%). Judged on success alone, they would look like the best robots.
- **Φ-slip, low friction (0.2–0.5).** It produced slip, but the robots could not walk properly at the nominal friction
  of 0.8 (83% and 0%).
- **Φ-slip, high friction only (1.0–1.2).** The robots walked well and did not slip beyond the clean range.
- **R-jitter.** Even at 3% of the stock action-rate weight, 3 of 4 robots stayed under the jitter threshold. P0 v5b
  robots, trained across payload, motor-strength and delay ranges, appear to stay smooth without the penalty doing
  much.
- **Φ-jitter.** One robot of two showed jitter. v1's no-delay robots (P2) were caught 3 of 3 by check 1's delay sweep,
  not by jitter. These pilots measured only nominal success and check 2, so delay fragility was not tested.
- **R-hop, × 10.** It changed nothing measurable.
- **R-hop, × 30.** It broke walking (59% and 2%) without the predefined hop signature. Whatever gait it produced, it
  was not flight at low speed.

The per-robot check 2 values (slip, jitter, flight share, vertical speed) are in the pilot bundle's `state.json`. They
are added here when the bundle is recorded.

## What happens next

Phase 2 as designed cannot test C2 or C3. Any revised design is a new, logged amendment approved by VJ. It must
disclose all 16 pilot robots above and be frozen before any new robot is trained.
