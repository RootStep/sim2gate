# v2 phase 2: conclusion (Oct 26, 2026; approved by VJ with amendments)

**In one line:** in a blinded study of 20 assigned simulated policy runs (Unitree Go2 in Isaac Lab), of which 19
completed measurement, ordinary training diagnostics identified the planted cause more often than behaviour alone, but
not by the preregistered margin (C2 not met), and ρ provided no demonstrated improvement over ordinary diagnostics in
this study (C3 not met). These are simulation results, still being validated; nothing here is about physical robots.

## What phase 2 asked

Can signals recorded **during training** tell **why** a trained policy has a problem, better than measuring the
finished policy? Two preregistered criteria (the frozen v2 plan; part B frozen Oct 25, SHA-256 `2d1b0a63…`):

- **C2:** training diagnostics (B) or training diagnostics plus ρ (C) classify the planted cause at least 20 points
  more accurately than behaviour measurements alone (A).
- **C3:** adding ρ improves on B by at least 10 points, or improves the early warning with a confidence interval
  above zero.

## How it was tested

- **Arms (after two pilot rounds):** control, R-slip (a weakened slip penalty: a reward cause), Φ-slip-mild (lower
  training friction: a physics cause of the same symptom) and Φ-delay (no command delay in training). Five policy runs
  each on fresh seeds 301–315, plus controls 101–105 from phase 1. The other planted problems tried in the pilots did
  not produce their intended symptom and were dropped; they stay in the record.
- **Blinding:** random IDs; the ID-to-arm key sealed to Vihaan's key; Vihaan ran the frozen scoring program, and the
  predictions' hash was recorded before he released the key.
- **Checks before the study:** the leak audit passed its specified checks for configuration leakage (those checks
  cannot establish the absence of every possible leak); an independent cross-check of the gradient maths with
  Vihaan's own package (756 directions agree to rounding error); and a documented review of the ρ derivation,
  authorised by VJ in place of Vihaan's pending sign-offs.

## The incomplete run, and which analysis is protocol-compliant

**Seed 313 (Φ-delay) never completed measurement.** Its training crashed twice at the same point with a PhysX GPU
error (about 1,730 of 2,000 rounds), so it was never measured. The frozen part B (section 4) says: *"If it fails
again, the robot is missing for every approach and the report says so."* The study driver did not follow this: it
exported the run's training features and the scoring program filled its missing behaviour features with
training-fold means, so the originally executed analysis scored 20 rows.

- **Protocol-compliant analysis (primary): 19 completed runs, seed 313 excluded.** Shown first below.
- **Originally executed analysis: 20 rows,** kept for transparency.

The compliant numbers are a recount of the frozen program's existing out-of-fold predictions without seed 313. The
frozen program cannot refit on 19 runs (its folds require five runs per arm, and part B did not define folds for a
missing run), so seed 313 still sat in the training folds of the other runs' models. **Neither acceptance verdict
changes under either analysis.**

## What it found

| | Behaviour only (A) | Training diagnostics (B) | B + ρ (C) |
|---|---|---|---|
| Cause identified, **protocol-compliant** (19 runs; chance 25%) | 17/19 (89%) | 19/19 (100%) | 19/19 (100%) |
| Cause identified, originally executed (20 rows) | 17/20 (85%) | 19/20 (95%) | 19/20 (95%) |
| The two slip arms told apart (R-slip vs Φ-slip-mild, 10 runs) | 8/10 | 10/10 | 10/10 |
| Final slip vs early-training prediction (Spearman, 19 measured runs) | — | 0.84 | 0.85 |

- **C2 not met.** B's margin over A was 10.5 points in the protocol-compliant analysis (10 points as originally
  executed), against the 20 required.
- **On the two slip arms,** training diagnostics classified all 10 runs correctly and behaviour 8. That is two more
  correct classifications among ten runs: encouraging, and a reason for another test, not a validated
  cause-diagnosis claim. The two arms differ substantially in slip severity, so this was not necessarily the hardest
  comparison.
- **C3 not met.** ρ provided no demonstrated improvement over ordinary diagnostics in this study: accuracy was
  unchanged, and the early-warning difference was uncertain (interval −0.05 to +0.08). ρ also ranked the runs exactly
  like the simple variance ratio, the same pattern as phase 0B: no demonstrated advantage over the variance baseline.
- **Early-training diagnostics were associated with final slip across this cohort** (rank correlation about 0.84 from
  the first 400 of 2,000 rounds, for every training-side approach). The pooled correlation may partly reflect
  differences between intervention groups rather than prediction of differences within each group.

## Limits

- **Small and narrow.** 20 assigned simulated policy runs (19 measured), one robot model, one task, three planted
  causes. A 10-point gap is two runs.
- **The arms were chosen after the pilots,** with knowledge of their results; disclosed throughout.
- **The incomplete run** (above): a deviation between the driver and the frozen protocol. Its crash repeats on the
  same seed, so the retry rule could not help.
- **Blinding rested partly on agreement.** The scoring computer is shared by VJ and Vihaan and holds his key; VJ did
  not look before the predictions were recorded.
- **Behaviour features can reveal some arms to a person looking** (R-slip slips about 25 times more). Only the frozen
  program used the labels.

## What it means for Sim2Gate

- **ρ is removed from current product claims.** It stays a research quantity in Vihaan's work, with phase 0B's and
  phase 2's limits attached.
- **Training-side diagnostics are a candidate for another test, not a validated feature.** Any product use needs a new,
  preregistered study on new runs, with causes matched more closely in symptom severity and within-group prediction
  examined.
- **Next, as decided before the study (amendment 1):** phase 3, fix verification (retrain with the recommended change
  against a matched retraining control, judged on held-out conditions), and phase 4, the blinded validation of the
  Layer 1 checks.

## Record

Protocol and freeze log: `PHASE2_PROTOCOL.md`, `PHASE2_FREEZE.md`. Scores and post-hoc findings:
`evidence/phase2/scoring/`. Pilots: `evidence/phase2/pilots/`, `evidence/phase2/pilots_amendment1/`. Leak audit:
`evidence/phase2/leak_audit/`. Gradient cross-check: `evidence/phase0b/crosscheck/`.
