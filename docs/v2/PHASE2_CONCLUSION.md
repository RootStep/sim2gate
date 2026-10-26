# v2 phase 2: conclusion (Oct 26, 2026)

**In one line:** in a blinded study on 20 simulated Go2 robots, ordinary training diagnostics identified the planted
cause better than behaviour alone, but not by the preregistered margin (C2 not met), and ρ added nothing over them
(C3 not met). These are simulation results, still being validated; nothing here is about real robots yet.

## What phase 2 asked

Can signals recorded **during training** tell **why** a robot has a problem, better than measuring the finished
robot? Two preregistered criteria (the frozen v2 plan, part B frozen Oct 25):

- **C2:** training diagnostics (B) or training diagnostics plus ρ (C) classify the planted cause at least 20 points
  more accurately than behaviour measurements alone (A).
- **C3:** adding ρ improves on B by at least 10 points, or improves the early warning with a confidence interval
  above zero.

## How it was tested

- **Arms (after two pilot rounds):** control, R-slip (a weakened slip penalty: a reward cause), Φ-slip-mild (lower
  training friction: a physics cause of the same symptom) and Φ-delay (no command delay in training). Five robots each,
  fresh seeds 301–315, plus controls 101–105 from phase 1. The other planted problems tried in the pilots did not
  produce their intended symptom and were dropped; they stay in the record.
- **Blinding:** random IDs; the ID-to-arm key sealed to Vihaan's key; Vihaan ran the frozen scoring program and the
  predictions' hash was recorded before he released the key.
- **Checks before the study:** a leak audit (no configuration value leaks into the inputs), an independent
  cross-check of the gradient maths with Vihaan's own package (756 directions agree to rounding error), and a
  documented review of the ρ derivation, authorised by VJ in place of Vihaan's pending sign-offs.

## What it found

| | Behaviour only (A) | Training diagnostics (B) | B + ρ (C) |
|---|---|---|---|
| Cause identified correctly (20 robots; chance 25%) | 85% | 95% | 95% |
| Slip pair told apart (reward vs physics cause, 10 robots) | 80% | 100% | 100% |
| Final slip predicted from the first 20% of training (Spearman) | — | 0.84 | 0.85 |

- **C2 not met.** B beat A by 10 points, not 20. Where the question is hardest (same symptom, different cause),
  training diagnostics were right every time and behaviour was not.
- **C3 not met.** ρ gave the same accuracy as B and no reliable early-warning gain (difference interval −0.05 to
  +0.08). It also ranked robots exactly like the simple variance ratio, the same pattern as phase 0B. So there is no
  demonstrated advantage over the variance baseline.
- **Early warning works in this setting.** The first 400 of 2,000 training rounds predicted each robot's final foot
  slip well (about 0.84), for every training-side approach.

## Limits

- **Small and narrow.** 20 simulated robots, one robot model, one task, three planted causes. A 10-point gap is two
  robots.
- **The arms were chosen after the pilots**, with knowledge of their results; disclosed throughout.
- **One robot was lost.** Seed 313 (Φ-delay) crashed twice at the same point with a PhysX GPU error, so it was never
  measured. Part B says it should have been left out entirely; the driver scored it with missing measurements. Leaving
  it out gives A 17/19 and B and C 19/19: neither verdict changes. The crash repeats on the same seed, so the retry
  rule could not help.
- **Blinding rested partly on agreement.** The scoring computer is shared by VJ and Vihaan and holds his key; VJ did
  not look before the predictions were recorded.
- **Behaviour features can reveal some arms to a person looking** (R-slip slips about 25 times more). Only the frozen
  program used the labels.

## What it means for Sim2Gate

- **ρ is not a product claim.** It stays a research quantity in Vihaan's work, with phase 0B's and phase 2's limits
  attached. Sim2Gate's reports do not use it to explain causes.
- **Training-side diagnostics are a promising candidate, not a validated feature.** They separated the reward and
  physics causes of slipping perfectly here and gave a useful early warning, but missed the preregistered bar. Any
  product use needs a new, preregistered test on new robots and more than one cause per symptom.
- **Next, as decided before the study (amendment 1):** phase 3, fix verification (retrain with the recommended change
  against a matched retraining control, judged on held-out conditions), and phase 4, the blinded validation of the
  Layer 1 checks on new robots.

## Record

Protocol and freeze log: `PHASE2_PROTOCOL.md` (frozen `2d1b0a63…`), `PHASE2_FREEZE.md`. Scores and post-hoc findings:
`evidence/phase2/scoring/`. Pilots: `evidence/phase2/pilots/`, `evidence/phase2/pilots_amendment1/`. Leak audit:
`evidence/phase2/leak_audit/`. Gradient cross-check: `evidence/phase0b/crosscheck/`.
