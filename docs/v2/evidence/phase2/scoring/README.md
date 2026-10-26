# Phase 2 study: stage 2 scores (Oct 26, 2026)

**Under the frozen rules: C2 not met, C3 not met.** Simulated robots with planted interventions (Unitree Go2,
Isaac Lab); 20 robots in the primary set (5 per label). These are simulation results, still being validated, and say
nothing yet about real hardware.

Every hash checked before scoring: predictions `3e8d4f5f…` (recorded at 15:03 ET, before the key release), feature
table `14e0feb9…`, scoring program `7231e36c…` (frozen Oct 25). Key released by Vihaan after that:
`key.json` SHA-256 `fc8bfeadbfa38d4d22097577d89b75cbb5ebaa410c6fde96acbdc26a6dd39c12`. Result file
`stage2_result.json`, SHA-256 `6226a2739b0dbac553fc35f7f7917b8cfdeafb0132afeecadaaaa098233cf71f`.

## Scored criteria

| Criterion | Rule | Result |
|---|---|---|
| **C2** | B's or C's accuracy at least 20 points above A's | **Not met.** A 0.85, B 0.95, C 0.95: 10 points |
| **C3** | C at least 10 points above B in accuracy, or C's early-warning Spearman above B's with the 95% paired-bootstrap interval above zero | **Not met.** Accuracy: C = B (0.95). Early warning: C 0.853, B 0.842, interval −0.054 to 0.084 |
| C against B-var (interpretation) | | No demonstrated advantage over the variance baseline: equal accuracy (0.95); identical early-warning ranking (interval 0 to 0) |

Chance accuracy is 0.25.

## Reported, not scored

| | A | B | C | B-var | C-placebo | B_w | C_w |
|---|---|---|---|---|---|---|---|
| Accuracy, 4 labels | 0.85 | 0.95 | 0.95 | 0.95 | 0.90 | 0.95 | 0.95 |
| Reward vs physics (15 non-controls) | 0.80 | 0.93 | 0.93 | 0.93 | 0.87 | 0.93 | 0.93 |
| Slip pair (R-slip vs Φ-slip-mild, 10 robots) | 0.80 | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 |
| Without Φ-delay (15 robots) | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Qualified only (19 robots; conditional on qualification) | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Early warning, slip (Spearman, window 1) | — | 0.842 | 0.853 | 0.853 | 0.842 | 0.863 | 0.846 |
| Early warning, latency cliff (Spearman) | — | 0.47 | 0.56 | 0.56 | 0.47 | 0.52 | 0.57 |

**Confusion.** Every approach classified all 5 controls and all 5 R-slip robots correctly. A labelled 2 Φ-slip-mild
robots (seeds 306, 307) as Control; B and C got all 5 right. All approaches labelled one Φ-delay robot (seed 313) as
Control.

## Found after unblinding (post hoc; nothing is rescored)

- **Seed 313 has no behaviour measurements.** Every A feature (walking, check 1, check 2, check 5) is missing in the
  feature table, while its training features are present. That is why it is the "unqualified" robot in the secondary
  analysis, and why it is missing from the early-warning correlations (19 pairs). The study printout reported counts
  only, and the cause has not yet been read from the sealed records. Part B says a run failure is retrained once and,
  if it fails again, the robot is missing **for every approach**. The driver instead exported its training features,
  so it entered scoring with behaviour features filled by training-fold means. This is a deviation between the driver
  and the frozen protocol, recorded here.
- **The deviation does not change either verdict.** Leaving seed 313 out of the counts: A 17/19 (0.895), B and C 19/19
  (1.00). That is 10.5 points, still short of C2's 20. C and B remain equal. (This is a count over the existing
  out-of-fold predictions, not a re-run of stage 1.)
- **The 268 missing training updates** (437,052 of 440,000 ρ values) probably belong to the same run, if its training
  stopped early. To be confirmed from the records.
- **ρ and the variance ratio gave the same early-warning ranking** on all 20 robots: their window-1 predictions differ
  by at most 0.034 (log units), never enough to swap two robots. The same pattern as phase 0B.
- **Behaviour alone was strong.** Contrary to the expectation recorded after the pilots, the Φ-slip-mild robots' slip
  (0.158–0.362 m/s) did not always separate them from controls in A's full feature set. Training signals separated
  them every time.
