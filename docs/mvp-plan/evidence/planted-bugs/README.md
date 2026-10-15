# Planted-bug test of the MVP validation (Oct 15, 2026)

The plan's "Can the validation fail?" step, done before the code freeze (step 3): plant one bug in a scratch copy of
each check and confirm the validation catches it. Reproduce with `python -m sim2gate.mvp.planted_bugs` (no Isaac
Sim needed); `tests/test_validation.py` runs it on every commit. `result.json` and `result.txt` are its output.

## What runs

The same pipeline the blinded run will use, end to end:

1. each check scores a policy's measurements: check 2 (`sim2gate.checks.exploits_score`, offline from the check 2
   report and `calibration/p0v4/thresholds.json`), check 1 and check 5 (with `calibration/p0v4/checks15/success.json`);
2. `sim2gate.mvp.verdict` turns the three results into go / no-go, with the plan's exit codes;
3. `sim2gate.mvp.acceptance` scores all verdicts against the answer key: A1–A6, with three replays for A6.

Each bug is a one-line edit to a copy of the check's source file, imported under its own name; the checks
themselves are never changed (the test asserts this afterwards). Two bugs per check: one that hides real flags,
one that raises false alarms.

| Direction | Bug | Edit (in the scratch copy) | Must break | Result |
|---|---|---|---|---|
| hides flags | check 2 thresholds 10x too high | `v > t` → `v > t * 10` | A1 | caught (A1 failed) |
| false alarms | check 2 thresholds 10x too low | `v > t` → `v > t / 10` | A4 | caught (A4 failed) |
| hides flags | check 1 cliff only below 8% success, not 80% | `CLIFF_SUCCESS = 0.8` → `0.08` | A2 | caught (A2 failed) |
| false alarms | check 1 flags any cliff in the sweep | drops the realistic-range condition | A4 | caught (A4 failed) |
| hides flags | check 5 flags falls only when fewer than 8% of Newton episodes are fall-free, not 80% | `NEWTON_FALL_FREE_MIN = 0.8` → `0.08` | A3 | caught (A3 failed) |
| false alarms | check 5 error-ratio limit 0.2, not 2 | `ERROR_RATIO_MAX = 2.0` → `0.2` | A4 | caught (A4 failed) |

Check 5 flags a policy when its Newton fall-free rate is *below* the minimum, so lowering the minimum hides falls:
that bug missed every stand-in P5 (A3 failed) and raised no false alarm (A4 held).

Control: unedited scratch copies of all three checks pass A1–A6 on the same set. The test can also say "not
caught": a harmless edit (error-ratio limit 2.0 → 2.5) leaves the validation passing (tested).

## The stand-in answer-key set

The plan's shape: 23 blinded IDs, P0 seeds 1–8 and P1–P5 seeds 1–3. It is a test fixture built from measured data,
not a result about any planted policy (those are trained after the freeze):

- **P0 1–8, real:** the P0 v4 calibration baselines' measurements (seeds 1, 2, 3, then 1, 2, 3, 1, 2 again).
  All eight get "go".
- **P2, real:** the P0 v2 policies, trained with no command delay, which is P2's planted flaw. Their raw files are
  in `p0v2/` (from `s2g-p0-v2-superseded.tgz`, SHA-256 `2ee77505…`; `p0v2/SHA256SUMS`), and their check 2 reports
  are in `calibration/p0v2/`. The real check 1 flags latency on all three seeds (cliffs at 20, 25, 30 ms).
- **P1, P3, P4, P5, synthetic:** each is a P0 v4 measurement with its target signal edited in. P1 fails every
  episode at friction 0.6 and below; P3's foot slip and P4's action jitter are doubled; P5 falls in 2 of every 5
  Newton episodes.

## Seen along the way (no rule changed)

- **P1 may be harder to detect than planned.** The P0 v2 policies were trained on a single friction value (0.8)
  and still walked down to friction 0.33: check 1 did not flag friction on any of the three seeds. P1 narrows
  friction and mass randomization to near zero; on flat ground, narrow friction alone may not put the cliff inside
  the realistic range. A2 can still pass on mass. A weakened training recipe is not the same as a demonstrably
  weak policy: if check 1 does not flag P1, A2 fails and is reported as failed. No threshold, range or rule is
  changed to make P1's label come true; the diagnosis then asks whether those P1 policies are in fact weak (for
  example, where their friction and mass cliffs lie) or robust despite the narrower training.
- **The v2 policies are development evidence only.** They stand in for P2 in this test because their flaw is P2's,
  but the final P2 policies are trained with the same recipe and budget as every other test policy (P0 v4), with
  only the delay held at 0 (C7).
- **Check 2 also flags the real v2 policies** on actuator limits (all three) and jitter (seed 3), because the v4
  thresholds come from smoother, randomized policies (`calibration/p0v4/observations.md`, item 3). For P2 these are
  non-target flags, which the plan reports but does not count against the checks.
