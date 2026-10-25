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

- **R-slip.** At 10% slip penalty both robots slid far beyond every clean robot, yet scored 99.6% nominal success.
  Under the same v2 threshold, the 20 clean robots scored 91.8–99.2% (`../../phase1/`, Watch record). So R-slip is at
  the top of the clean range, not clearly above it: 0.996 against 0.992 is one episode in 256. Judged on success alone,
  these robots would look as good as the best clean ones. Corrected Oct 24, after VJ's review: an earlier draft
  compared them with the clean range under v1's threshold.
- **Φ-slip, low friction (0.2–0.5).** It produced slip, but both robots failed the success test at the nominal friction
  of 0.8 (83% and 0%) with **zero falls**: poor task performance (tracking), not falling.
- **Φ-slip, high friction only (1.0–1.2).** The robots walked well and did not slip beyond the clean range.
- **R-jitter.** Even at 3% of the stock action-rate weight, 3 of 4 robots stayed under the jitter threshold. P0 v5b
  robots, trained across payload, motor-strength and delay ranges, appear to stay smooth without the penalty doing
  much.
- **Φ-jitter.** One robot of two showed jitter. v1's no-delay robots (P2) were caught 3 of 3 by check 1's delay sweep,
  not by jitter. These pilots measured only nominal success and check 2, so delay fragility was not tested.
- **R-hop, × 10.** It changed nothing measurable.
- **R-hop, × 30.** It broke walking (59% and 2%) without the predefined hop signature. Whatever gait it produced, it
  was not flight at low speed.

## Measured values (check 2, play mode)

From the pilot's `state.json` (copied here; SHA-256 below).

| Robot | Success | Slip p95 (m/s) | Jitter share | Actuator share | Flight share, low speed | Vertical RMS, low speed (m/s) |
|---|---|---|---|---|---|---|
| R-slip_k0_1_s201 | 0.996 | 2.831 | 0.01079 | 0.0025 | 0.0022 | 0.073 |
| R-slip_k0_1_s202 | 0.996 | 2.692 | 0.01241 | 0.0029 | 0.0024 | 0.065 |
| Phi-slip_low_s205 | 0.828 | 0.476 | 0.00840 | 0.0037 | 0.0020 | 0.085 |
| Phi-slip_low_s206 | 0.000 | 0.174 | 0.01085 | 0.0043 | 0.2555 | 0.125 |
| Phi-slip_high_s207 | 0.973 | 0.101 | 0.00905 | 0.0035 | 0.0020 | 0.070 |
| Phi-slip_high_s208 | 1.000 | 0.102 | 0.00819 | 0.0032 | 0.0020 | 0.062 |
| R-jitter_k0_1_s209 | 0.977 | 0.108 | 0.00987 | 0.0035 | 0.0020 | 0.057 |
| R-jitter_k0_1_s210 | 0.984 | 0.112 | 0.00973 | 0.0035 | 0.0020 | 0.052 |
| R-jitter_k0_03_s211 | 0.984 | 0.112 | 0.00848 | 0.0039 | 0.0020 | 0.066 |
| R-jitter_k0_03_s212 | 0.992 | 0.115 | 0.00977 | 0.0031 | 0.0022 | 0.067 |
| Phi-jitter_nodelay_s213 | 0.992 | 0.093 | 0.00969 | 0.0035 | 0.0018 | 0.052 |
| Phi-jitter_nodelay_s214 | 1.000 | 0.092 | 0.01172 | 0.0048 | 0.0020 | 0.052 |
| R-hop_m10_s215 | 0.910 | 0.103 | 0.01051 | 0.0067 | 0.0022 | 0.074 |
| R-hop_m10_s216 | 0.984 | 0.104 | 0.01117 | 0.0077 | 0.0020 | 0.085 |
| R-hop_m30_s217 | 0.586 | 1.815 | 0.01070 | 0.0800 | 0.0032 | 0.109 |
| R-hop_m30_s218 | 0.016 | 0.033 | 0.01963 | 0.0208 | 0.0028 | 0.091 |
| Controls 101–105 (maximum) | — | 0.106 | 0.00925 | 0.0027 | 0.0020 | 0.060 |
| Symptom threshold | ≥ 0.85 to qualify | > 0.118 | > 0.00986 | — | ≥ 0.02 | ≥ 0.090 |

What the numbers add, reported only:

- **R-slip.** Slip was about 25× the clean maximum (2.7–2.8 m/s), and both robots were also above the jitter threshold
  (0.0108 and 0.0124).
- **R-jitter.** Weakening the action-rate penalty barely moved jitter. Values were 0.0085–0.0099, inside or at the edge
  of the clean range (controls up to 0.0092). Seed 209's "symptom" cleared its threshold by 0.00001.
- **Jitter came from elsewhere.** R-slip, R-hop × 10 and Φ-jitter seed 214 all had higher jitter than any R-jitter
  robot. In this recipe, check 2's jitter metric responds to other changes more than to the action-rate penalty.
- **Φ-slip low, seed 206.** It is the one robot that met the hop measures: 25.5% of low-speed steps airborne, and
  vertical RMS 0.125 m/s. It qualified on neither walking (0%) nor the arm it was planted for.
- **R-hop × 10.** It raised actuator saturation 2–3× above the controls (0.0067–0.0077 against 0.0027), but produced no
  flight.
- **R-hop × 30, seed 217.** Slip 1.8 m/s, actuator share 0.080 (about 30× the controls) and vertical RMS 0.109, but no
  flight.
- **R-hop × 30, seed 218.** Barely moved: 43 falls in 256 episodes, and jitter 0.0196.
- **Most interventions changed several metrics at once.** The plan's assumption of one symptom per planted cause does
  not hold well for this recipe. That matters for any redesign.

`state.json` SHA-256: `ce1ee52d337d68178f301c7bc2b57778ccffddb0492aeaee951956222008fe02`. The full pilot bundle (training records and checkpoints) is kept outside the
repository: `phase2-pilot.tgz`, 1184 files, 1887 MB, SHA-256
`6b823bc2249596701ec4f335c3f81c5d68db8504ae34c606f90df881dee589c5` (made Oct 24 on `sim2gate-arena-2`).

## What happens next

Phase 2 as designed cannot test C2 or C3. VJ approved a bounded final pilot round on Oct 24 (`PHASE2_PART_A_AMENDMENT1.md`).
It discloses all 16 pilot robots above.

Matching reward-term lists and clean configuration checks close the missing-term leak only. They do not show that
weighted log magnitudes cannot reveal an intervention; part B's leak audit tests that.
