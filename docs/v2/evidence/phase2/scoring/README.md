# Phase 2 study: stage 2 scores (Oct 26, 2026)

**Under the frozen rules: C2 not met, C3 not met.** 20 assigned simulated policy runs with planted interventions
(Unitree Go2, Isaac Lab), of which 19 completed measurement; see "Protocol-compliant analysis" below. These are simulation results, still being validated, and say
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
  analysis, and why it is missing from the early-warning correlations (19 pairs). **Cause (read from the unsealed
  records after stage 2):** training ran twice. The first attempt's folder was set aside at 11:30 UTC
  (`study_s313.incomplete-20261026113050`). The second crashed after about 2,500 s with a PhysX GPU error ("CUDA error:
  device-side assert triggered", in PhysX's GPU memory allocator and articulation views), leaving no final checkpoint,
  so nothing was measured. The first attempt had crashed the same way, at about the same point (2,545 s against
  2,503 s; roughly 1,730 of 2,000 rounds). Under part B that is a run failure, not an outcome.
- **The crash is reproducible for this seed.** A retry with the same seed replays the same simulation, so it hit the
  same fault. Whether the fault comes from the planted change (no command delay) driving the simulation into an
  unstable state, or from PhysX itself, is not known; no pilot or other study robot crashed. Lesson for later
  protocols: a run-failure retry on the same seed cannot help with a deterministic crash, so the rule should name a
  replacement seed or record the crash as a reported outcome. Part B says a run failure is retrained once and,
  if it fails again, the robot is missing **for every approach**. The driver instead exported its training features,
  so it entered scoring with behaviour features filled by training-fold means. This is a deviation between the driver
  and the frozen protocol, recorded here.
- **Protocol-compliant analysis (designated Oct 26, approved by VJ).** Part B section 4 (frozen text, SHA-256
  `2d1b0a63…`, checked) requires seed 313 to be missing for every approach. The protocol-compliant result is therefore
  the recount without it: A 17/19, B 19/19, C 19/19 (B-var 19/19, C-placebo 18/19, B_w and C_w 19/19); reward vs
  physics A 12/14, B and C 14/14. The 20-row scores above are the originally executed analysis, kept for transparency.
  The recount uses the frozen program's existing out-of-fold predictions; the program cannot refit on 19 runs (its
  folds require five per arm: "Phi-delay: 4 robots in the key, expected 5"), and part B did not define folds for a
  missing run. **Neither verdict changes:** C2's margin is 10.5 points against 20; C equals B.
- **The 268 missing training updates** (437,052 of 440,000 ρ values) probably belong to the same run, if its training
  stopped early. To be confirmed from the records.
- **ρ and the variance ratio gave the same early-warning ranking** on all 20 robots: their window-1 predictions differ
  by at most 0.034 (log units), never enough to swap two robots. The same pattern as phase 0B.
- **Behaviour alone was strong.** Contrary to the expectation recorded after the pilots, the Φ-slip-mild robots' slip
  (0.158–0.362 m/s) did not always separate them from controls in A's full feature set. Training signals separated
  them every time.
