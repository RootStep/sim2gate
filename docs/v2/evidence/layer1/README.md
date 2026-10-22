# v2 Layer 1 limits: results (Oct 22, 2026)

The limits were computed once from the 20 clean P0 v5b robots by `sim2gate.v2.layer1_limits` (commit `5d1696a`),
under the protocol frozen as `LAYER1_LIMITS_PROTOCOL.md` (SHA-256 `3b72a485…`, approved by VJ). The check 2 and Newton
values were not read before the freeze.

**Input.** `phase1-clean-measurements.tgz` (SHA-256 `9e1be1705da1073e1bd5a68cc2fe5469846ead13cae5663390b4d081be5fb38e`,
1.4 MB). It holds `MANIFEST.json` and the 82 measurement files extracted from `phase1-clean.tgz` (`36bb9b75…`) on VJ's
PC. Every file matched its SHA-256 in the manifest, and every measurement matched its robot's checkpoint. VJ confirmed that the bundle on his PC still has SHA-256 `36bb9b75…` and that the extract has `9e1be170…`, which is
the same as the attached file.

## The limits

| Limit | Value | How it was set |
|---|---|---|
| v2 episode success | Mean tracking error at most **0.1187 m/s**, and no fall | 1.5 × the clean reference of 0.0791 m/s, the mean of the 20 robots' means. Per-robot means were 0.072–0.086 m/s. 3 falls were left out of 5,120 nominal episodes. v1's threshold was 0.1133 m/s |
| Combined check 2 | Flag when the worst metric's ratio to the clean maximum exceeds **1.0541** | Highest of the 20 clean scores (seed 122; actuator share drove it). No floor was needed |

Clean maxima, which a new robot's metrics are divided by:

| Metric | Clean maximum |
|---|---|
| Foot slip p95 | 0.1124 m/s |
| Ground penetration | 0.00352 m |
| Actuator-limit step share | 0.00366 |
| Action-jitter high-frequency share | 0.00935 |

Contact force is reported only.

Files: `frozen_success_p0v5b.json` and `check2_limit_v2.json`. Copies used by later phases are in `src/sim2gate/v2/`.

## Clean robots under the frozen rules

| Check | Clean robots flagged | 95% interval |
|---|---|---|
| Check 1, as recorded (v1 threshold) | 1 of 20 (seed 122, sensor noise) | 0.01–0.24 |
| Check 2, leave-one-out | 0 of 20 | 0.00–0.16 |
| Check 5 | 0 of 20 | 0.00–0.16 |
| Any no-go | 1 of 20 | 0.01–0.24 |

**Stop rule (section 5):** not triggered. There was 1 no-go; the rule stops only if more than 3 of 20 get one.

## Observations, reported only (no limit changes)

- **Check 1 under the new success threshold.**
  - The new threshold is slightly looser than v1's, and under it no clean robot is flagged (0 of 20). Seed 122's
    sensor-noise cliff moves out of the realistic range.
  - Test robots will be scored under the new threshold. The recorded 1 of 20 is the count the stop rule uses.
- **Check 2 scores are tightly grouped.**
  - Clean scores run from 0.88 to 1.05, so the limit sits only about 5% above the worst clean metric. That suits a mild
    flaw such as v1's missed jitter robot.
  - It also means a new clean robot only slightly outside the clean range would be flagged. The construction keeps that
    to at most 1 in 21. Leave-one-out flagged 0 of 20, with per-robot limits from the other 19 between 1.05 and 1.23.
  - Foot slip drove most scores (14 of 20), jitter 4, and penetration and actuator share 1 each.
- **Check 5 is far from its limits.**
  - Every clean robot stayed upright in at least 99.6% of Newton episodes.
  - Tracking-error ratios (Newton / PhysX) were 1.32–1.61, against a limit of 2.
  - Newton success rates of 43–75% are reported only, in line with v1's clean robots.
