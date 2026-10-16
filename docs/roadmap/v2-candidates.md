# Sim2Gate version 2: candidate changes

Ideas found while building and validating the MVP. None of them changes the MVP: its rules are frozen for the blinded
validation. Each one needs its own evidence before it becomes a rule.

## 1. Joint sweep as a real check (found Oct 16, 2026)

**Finding.** The clean P0 v4 robot (seed 1) passes check 1: tested one condition at a time, it succeeds 91–98% of the
time across every realistic range. In the joint sweep, where all five conditions are drawn together, it succeeds in
only **74%** of the 504 episodes in which every condition is realistic at once. Some pairs are much worse together
than either alone (both in the harder half of their realistic range):

| Pair | Success together | Expected if independent |
|---|---|---|
| Extra base mass + weaker motors | 47% | 60% |
| Action latency + observation noise | 67% | 82% |
| Low friction + action latency | 74% | 83% |

**Why it matters.** One-condition-at-a-time testing can make a policy look safer than it is: on a real robot,
conditions stack. This is a differentiator worth building on.

**What version 2 would need.**
- A flag rule fixed in advance, e.g. "success with every condition realistic at once below X%", with X calibrated
  on clean baselines the way check 2's thresholds are.
- Evidence on more policies that the joint result separates weak from robust policies better than the
  one-at-a-time sweep does (the MVP's blinded run records the joint sweep for all 23 policies, report-only).
- Arena's simulation-based sensitivity analysis on the saved per-episode tables, for interactions beyond pairs.

Source: `mvp_runs/p5_pilot/joint_smoke.json` on `sim2gate-arena-2` (bundle `s2g-p5pilot-2026-10-16.tgz`).

## 2. Check 2 thresholds depend on the training recipe

The P0 v4 robots (trained with friction and delay randomization) are much smoother than P0 v2. The actuator-limit and
jitter thresholds calibrated on them would flag P0 v2 on those alone (`calibration/p0v4/observations.md`, item 3).
Version 2 could calibrate thresholds per training-recipe family, or report check 2 relative to the policy's own
recipe.

## 3. Newton run-to-run variation

PhysX results reproduce exactly; Newton runs vary slightly (one episode in 64 can flip between falling and not, error
ratio ±0.02; `evidence/pre-freeze/README.md`). Version 2 could run check 5's Newton side more than once, or with
more episodes, and report the spread.

## 4. Port to Isaac Lab-Arena

Planned since C2: checks 1, 2 and 5 as Arena variation and metric plug-ins, after the MVP validation.
