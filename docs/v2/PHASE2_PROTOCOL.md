# v2 phase 2: protocol (part B DRAFT; part A frozen separately in PHASE2_PART_A.md)

Phase 2 is the cause study (v2 plan, "Layer 2"). It tests two things on robots with planted interventions:

- whether training-side signals classify the planted intervention better than the finished robot's behaviour (C2);
- whether ρ adds to ordinary training diagnostics (C3).

The output is a **classification of the planted intervention**, not proof of a unique cause. Each planted change is a
hypothesis about what it does, and the pilots check it.

Revision 2 takes in VJ's review of Oct 22 (decision 1 approved in principle, with safeguards).

**Two freezes.**

| Part | Sections | Frozen when |
|---|---|---|
| A | 2–3: arms, candidate settings, pilot rules, leak audit | Before any pilot robot is trained |
| B | 4–8: study robots, inputs, ρ, rules, blinding, scoring | After the pilots and the leak audit, after Vihaan's two sign-offs, and before any study robot |

Each part's SHA-256 is recorded in `PHASE2_FREEZE.md` when it is frozen.

## 1. Preconditions

| Item | Status |
|---|---|
| Phase 1: 20 clean P0 v5b robots with full training records | Done Oct 22 |
| Layer 1 limits (success threshold 0.1187 m/s; combined check 2 limit) | Done Oct 22 |
| C1: Vihaan confirms the Oct 9 RSL-RL acceptance gate | **Pending**, needed for part B |
| Phase 0B: Vihaan's sign-off on Q1 (per-term ρ) and Q4 | **Pending**, needed for part B |
| C3 kept, with no extra compute | Decided by VJ, Oct 22 |

## 2–3. Arms, pilots and leak audit (part A)

Frozen separately, in `PHASE2_PART_A.md` (Oct 23, 2026).

## 4. Study robots (part B)

- **Seeds and recipe.** Each kept arm trains 5 robots on seeds 301–325, with the full training record
  (EvidenceRecorder, as phase 1).
- **Every robot is fully measured, qualified or not:**
  - nominal PhysX over 256 episodes;
  - check 1 on the v2 envelope;
  - check 2 with the hop measures;
  - Newton.
- **Failures are kept.** A robot that does not qualify is replaced by the next spare seed (326–345), and still stays in
  the record and in the secondary analysis.
- **Analysis sets.**
  - Primary: the first 5 qualified robots of each arm, by seed order.
  - Secondary: every trained robot.
  - Failures to qualify are reported per arm as an outcome of the intervention.
- **Stop rule.** More than 3 failures in one arm stops the run for a decision.
- **Controls.** The 5 control robots (101–105) were not used to develop any attribution rule; the rules in section 7
  are fixed without data. They are re-measured under phase 2's measurement code, so that every robot's behaviour
  features come from the same code version. A measurement is reused only where its code is unchanged, which is checked
  by version hash. Their phase 1 training records (TensorBoard and recorder) give B and C what they need.
- **GPU estimate.**
  - 25 study robots: about 23 hours.
  - Re-measuring the controls: about 2 hours.
  - Full measurement of failed robots: about 0.4 hours each.

## 5. Inputs and leakage (part B) ★

**Primary input set.** It contains only measured quantities:

- the behaviour measurements (A);
- for B and C, the **unweighted** per-term reward components: each term's own value per step, before its weight. The
  export step computes them while it still has the configuration, and only the unweighted values leave that step.

**Excluded from primary inputs:** weighted contributions, reward weights, run folder names, task names, saved configs,
seeds and file names.

**Secondary experiment, reported separately.** B_w and C_w are B and C plus the weighted contributions. They test
diagnosis with reward-design clues available. They are never pooled with the primary result.

**Export.** One frozen exporter turns each robot's records into a feature table under a random ID. It is tested on
synthetic data to confirm:

- unweighted components do not change when a weight is rescaled while behaviour is held fixed;
- no excluded field reaches the output.

## 6. What ρ means here (part B) ★

C uses the per-term quantity from phase 0B: ρ_k = corr(A, A − T_k).

- **A:** the trainer's unnormalised GAE advantages for the rollout, from the total reward and the critic in use at
  that update.
- **T_k:** term k's reward tail, the same GAE recursion applied to term k's weighted rewards with zero values, so that
  A − T_k is the advantage with term k removed and the critic held fixed.
- **Computation:** from the recorder's per-update covariance of [A, T_1…T_K, V-part], with the closed form verified
  numerically in phase 0B.
- **Normalisation:** correlation is unchanged by RSL-RL's per-rollout advantage normalisation (subtract the mean,
  divide by a positive scale), so ρ_k is computed on unnormalised values.
- **Status:** this is **an extension beyond V1.4**, which concerns bootstrapped reward shaping. Phase 0B gave a
  provisional answer under its frozen rules, with recorded limits:
  - a variance ratio ranked terms almost as well;
  - ρ did not track the foot-slip term within runs.

  Part B is not frozen until Vihaan signs off the derivation (Q1).
- **ρ for a weakened term:** it measures how much the term still steers the update, which is the diagnostic idea. The
  term is never zero, so ρ_k = 1 by construction does not arise.
- **Undefined values:** ρ_k is undefined for an update when Var(A) or Var(A − T_k) is below 10⁻¹² × Var(A) of the same
  update. Undefined updates are left out of the window average. If more than half of a window's updates are undefined,
  that feature is missing (section 7).
- **What C contains:** C is B plus ρ_k only. It does not include β_k, R_k, the variance ratio or any other phase 0B
  quantity. A comparison that adds those would be named separately.
- **ρ carries magnitude information that B's primary inputs exclude.** T_k is built from the *weighted* term, so ρ_k
  partly reflects how strongly a term is weighted. A C-over-B gain could therefore come from that magnitude rather
  than from ρ's specific form. Section 8 controls for this with B-var: B plus Var(T_k)/Var(A) per term per window. B-var
  uses the same weighted information and was the baseline that nearly matched ρ in phase 0B.

## 7. Approaches, rules and cross-validation (part B) ★

**Windows.** Rounds 0–399, 400–999 and 1000–1999. Window 1 is the first 20% of training.

| Approach | Features per robot |
|---|---|
| A, behaviour only | Nominal success (v2 threshold); check 2's four scored metrics, contact force, flight share at low speed and vertical-speed RMS; check 1 cliff index for each of the five conditions (0–9; 10 = no cliff); check 5 fall-free share and error ratio |
| B, ordinary training diagnostics | For each reward term, the unweighted component averaged per window; action std, value loss and learning rate per window (training logs); KL and clip fraction per window (recorder) |
| C, B plus ρ | Exactly B's features, plus ρ_k per term per window |
| B-var (comparison for C3's interpretation) | B plus Var(T_k)/Var(A) per term per window: the same number of extra features as C, from the same weighted information |
| C-placebo (reported only) | B plus the same number of standard-normal noise features (seed 0) |

**B and C are identical except for ρ.** Same features, preprocessing, model and settings, and no tuning for either.
C-placebo shows how much of any C-over-B gain comes from having more features alone.

**Folds.** The unit is the independently trained robot. Fold *i* (1–5) holds each arm's *i*-th robot, and fold 1 also
holds control robot 101. Episodes and log rows are never split across folds.

**Preprocessing (training folds only).**

1. Z-score each feature with the mean and standard deviation of the training folds.
2. A feature with zero standard deviation in the training folds is dropped for that fold.
3. A missing value is set to the training-fold mean (0 after z-scoring) and counted.
4. A feature missing for more than 20% of robots is dropped for every approach, and reported.

**Question 1, classification of the planted intervention.**

- **Rule:** nearest centroid, Euclidean distance, equal feature weights.
- **Ties:** go to the first label in the order Control, R-slip, Φ-slip, R-jitter, Φ-jitter, R-hop.
- **Score:** accuracy over all robots of the primary set, with the reward-vs-physics accuracy also reported.

**Question 2, early warning.**

- **Inputs:** window 1 features only.
- **Rule:** ridge regression, penalty 1.0, predicting the log of each final symptom measure separately: foot slip p95,
  jitter share, and flight share at low speed.
- **Score:** Spearman correlation between predicted and measured values over all primary robots, averaged over the
  three.
- **A is excluded:** it has no training-time data.

## 8. Blinding, labels and scoring (part B) ★

- **Random IDs.** The driver gives every study robot and every control a random ID.
- **The key.** The ID-to-arm key is encrypted to Vihaan's public key, as in v1. The plain key is not kept on the
  instance or in the repository.
- **Feature tables.** The exporter writes them under random IDs. Their SHA-256 hashes and the frozen scoring code are
  committed.
- **Sealed scoring, stage 1 (Vihaan, on his computer).** He runs the frozen scoring program with the key. For each fold
  it uses only the four training folds' labels and never reads the test fold's labels. It writes predictions per
  robot ID for every approach. He sends the predictions file and its SHA-256.
- **Stage 2 (after the predictions hash is recorded).** Vihaan releases the key. Stage 2 checks the hashes and computes
  the scores.
- **Who stays blind.** VJ and Claude see no labels until the predictions are committed.

**Criteria (as the frozen plan).**

| Criterion | Passes if |
|---|---|
| C2 | B's or C's accuracy is at least 20 points above A's |
| C3 | C's accuracy is at least 10 points above B's; or C's early-warning Spearman is above B's, with the 95% bootstrap interval of the difference above zero |

The bootstrap uses 10,000 resamples of robots, stratified by arm, with seed 0.

**Interpreting C3.** C3 is scored exactly as the plan says, C against B. The gain is attributed to **ρ specifically**
only if C also beats B-var under the same rule. If C beats B but not B-var, the report says the gain comes from the
size of each term's influence, which a simple variance ratio also captures, and not from ρ.

**Reported, not scored:**

- confusion matrices;
- per-symptom early warning;
- C-placebo and C against B-var;
- B_w and C_w;
- the secondary analysis set;
- results without Φ-jitter. Its change only starts at round 500, so window 1 cannot see it.

**Arms dropped in the pilots.** Scores are computed on the remaining arms, and the reduced design is stated. If two or
more flawed arms are dropped, C2 and C3 are reported as not tested.

## Decisions for VJ (★ = Vihaan reviews too)

- **D1 ★** Approved in principle. The safeguards are in sections 2, 3 and 5:
  - same terms, order and format;
  - unweighted components as the primary input;
  - weighted contributions only in a separate experiment;
  - a leak audit before part B is frozen.
- **D2.** Controls are seeds 101–105. They were not used for rule development, and they are re-measured under phase
  2's code.
- **D3.** Pilots run under part A: candidate settings, symptom thresholds and stopping rules fixed before the first
  pilot; separate seeds; every pilot reported.
- **D4 ★** Rules as in section 7: features, scaling, model, ties, missing data, robot-level folds and training-fold-only
  preprocessing.
- **D5.** Abnormal hopping needs flight share at low commanded speed, vertical motion and continued qualification
  together.
- **D6 ★** Sealed two-stage scoring. Vihaan runs stage 1 with training-fold labels only; VJ and Claude stay blind until
  the predictions are committed.
- **D7.** Every robot is fully measured and kept; failures are reported per arm; there is a secondary analysis on all
  robots.
- **D8 ★** ρ_k is defined as in section 6 and labelled an extension beyond V1.4; C differs from B only by ρ_k. A gain
  is credited to ρ only if C also beats B-var, which carries the same weighted-magnitude information.
