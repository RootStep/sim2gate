# v2 Layer 1 limits: protocol (frozen; approved by VJ, Oct 22, 2026)

This file says how Layer 1's v2 limits are computed from the 20 clean P0 v5b robots (phase 1). It is frozen, with its
SHA-256 recorded in a freeze file, **before** anyone reads those robots' check 2 or Newton values, and before any test
robot is trained (v2 plan, "Layer 1: setting the limits"; `PHASE1_PROTOCOL.md` section 5). Read so far: only each
robot's nominal success and check 1 flags (printed by the run), and seed 101's torque-clipped share (to confirm the
check 2 fix).

## 1. Data

- **Calibration set:** the 20 qualified robots: seeds 101–105, 107–111, 113–122. Seeds 106 and 112 did not qualify
  and are not used.
- **Bundle:** `phase1-clean.tgz`, SHA-256 `36bb9b75caf2ab7d37482bf7dc79c5ddd48cacfd2bde3f3aac834ef5c23b712a`.
  - The analysis reads the measurement files and the bundle's `MANIFEST.json`, extracted from that bundle.
  - It checks every file against the manifest's SHA-256.
  - It checks each measurement's checkpoint hash against the run state recorded in the manifest.
- **Measurements used:**
  - nominal PhysX (256 episodes);
  - check 2 (play mode, as recorded);
  - nominal Newton (256 episodes);
  - check 1 (v2 envelope).

## 2. v2 success threshold

- **Rule (v1's, unchanged):** an episode succeeds when the robot does not fall and its mean speed-tracking error is at
  most 1.5 × the clean reference error.
- **Reference:** for each robot, the mean tracking error over its nominal PhysX episodes that did not fall. The
  reference is the mean of those 20 per-robot values, so each robot counts equally.
- **Falls:** fallen episodes are left out of the reference. Each robot's fall count is reported.
- **Use:**
  - The threshold applies to every later measurement: test robots, phase 2–4 arms and fix verification.
  - It does not re-judge the 20 clean robots, which qualified under v1's threshold (`PHASE1_PROTOCOL.md` section 3).
  - It is written once to `frozen_success_p0v5b.json` and never overwritten.

## 3. Combined check 2 limit

**Metrics.** Four metrics are scored:

- `foot_slip_p95_mps`
- `ground_penetration_m`
- `actuator_limit_step_share`
- `action_jitter_high_freq_power_share`

`contact_force_p99_bw` is reported only (v2 plan: contact-force spikes move to v3). Higher is worse for all four.

**Score for a clean robot.** For robot *r* and metric *m*, divide its value by the highest value among the other 19
robots:

```latex
\text{ratio}_{r,m} = \frac{x_{r,m}}{\max\big(\max_{j \ne r} x_{j,m},\ f_m\big)}, \qquad s_r = \max_m \text{ratio}_{r,m}
```

The limit is the highest of the 20 scores: L = max over r of s_r.

**Score for a new robot.** The same ratio, against the highest value among all 20 clean robots. The robot is flagged
when its score is above L. The report names the metric with the largest ratio as the one that drove the flag.

**Floors.** f_m stops division by zero when every clean robot measures 0 on a metric:

| Metric | Floor |
|---|---|
| Slip | 0.001 m/s |
| Penetration | 0.0001 m |
| Actuator share | 0.0001 |
| Jitter share | 0.0001 |

The analysis reports whether any floor was used.

**Why at most 1 in 21.** Each clean robot's score is measured against 19 others, a new robot's against 20, so the
limit can only be higher than if all 21 robots were treated alike. If a new clean robot is exchangeable with the 20,
its chance of exceeding the limit is therefore at most 1 in 21. A simulation of the formula on Oct 17 gave 3–4%.

**Applicability (as in v1).** The limit applies only to check 2 reports measured with the same settings as the
calibration reports: measurement settings, contact sampling, physics path, command set and steps. Any other report is
"invalid", not scored.

## 4. Checks 1 and 5: rules unchanged, clean-robot rates reported

Both rules come from the frozen v2 plan and are not recalibrated here.

| Check | Rule | Reported on the 20 clean robots |
|---|---|---|
| Check 1 | Flagged when a cliff (success below 80%) falls inside a condition's realistic range | Flag count with a 95% interval: 1 of 20 so far (seed 122, sensor noise) |
| Check 5 | Flagged when fewer than 80% of Newton episodes are fall-free, or Newton's tracking error is more than 2× PhysX's | Each robot's fall-free share and error ratio, and the flag count with a 95% interval |

Check 1 cliffs are scored with the new success threshold for test robots. For the clean robots, they are also
re-reported under it, for information only.

## 5. Pre-declared stop rule (proposed)

Before any test robot is trained, count the clean robots that would get a no-go from any check: check 1 as recorded,
check 2 by leave-one-out (section 6), and check 5.

If more than 3 of the 20 would get a no-go, the run stops for a decision. The reason is B4: at most 1 of 5 held-out clean robots
may get a no-go. If 3 of 20 clean robots get a no-go (15%), B4 still passes about 84% of the time by chance. At
5 of 20, that drops to about 63%. A stop does not change any limit after the fact. Any
change is a logged amendment, tested on new robots.

## 6. Validation, reported only

**Leave-one-out.** For each clean robot, compute the limit from the other 19 (each scored against 18) and score the
left-out robot against it. Report how many of the 20 are flagged. This is the readiness gate "validate the check 2
calibration formula on held-out clean simulations". It is a check of the formula, not a new rule. The real test is B4
in phase 4, on 5 new clean robots.

## 7. Order of work

1. VJ approves this file. It is frozen and its SHA-256 recorded.
2. The analysis code is committed with tests on synthetic data, before it reads the bundle.
3. It runs once on the bundle. It writes:
   - `frozen_success_p0v5b.json`;
   - `check2_limit_v2.json`;
   - the clean-robot report for checks 1, 2 and 5.
4. Anything found afterwards is reported separately as exploratory and never changes these limits.

## 8. Not set here

- The held-out conditions for B7.
- The P1 development pilot.
- The test robots.

Each is set in phase 4's own protocol, frozen before those robots are trained.

## Decisions approved by VJ (Oct 22, 2026)

1. Keep v1's success rule: no fall, and error at most 1.5 × the clean reference.
2. Score four check 2 metrics, with contact force reported only.
3. Use the ratio-to-others'-maximum form of the combined score, with the floors above.
4. Stop for a decision if more than 3 of 20 clean robots would get a no-go.
5. Report leave-one-out validation without changing the rule.
