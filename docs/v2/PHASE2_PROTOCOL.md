# v2 phase 2: protocol (part B DRAFT; part A frozen separately in PHASE2_PART_A.md)

Phase 2 is the cause study (v2 plan, "Layer 2"). It tests two things on robots with planted interventions:

- whether training-side signals classify the planted intervention better than the finished robot's behaviour (C2);
- whether ρ adds to ordinary training diagnostics (C3).

The output is a **classification of the planted intervention**, not proof of a unique cause. Each planted change is a
hypothesis about what it does, and the pilots check it.

Revision 2 takes in VJ's review of Oct 22 (decision 1 approved in principle, with safeguards).

**Revision 4 (Oct 25, draft for VJ).** It follows the amendment 1 pilots, in which both new arms passed, and VJ's
direction of Oct 25:

- pilot tuning ends; there are no further pilot rounds;
- the amended study is frozen, then evaluated on fresh seeds;
- the original failed interventions stay in the record;
- pilot robots are excluded from confirmatory scoring;
- no-delay training is a separate intervention, delay fragility, not a jitter cause.

The arms, folds, labels and early-warning targets below are changed for that. Every change was made **after** the
pilots, knowing their results, and is disclosed as such.

**Revision 4, review corrections (Oct 25).** A documented review of the phase 0B files and this draft, authorised by
VJ in place of Vihaan's pending personal sign-offs, approved the per-term formula with stated assumptions and asked
for bounded corrections. They are made in sections 4, 6, 7 and 8 and listed at the end ("Review corrections").

**Two freezes.**

| Part | Sections | Frozen when |
|---|---|---|
| A | 2–3: arms, candidate settings, pilot rules, leak audit | Before any pilot robot is trained |
| B | 4–8: study robots, inputs, ρ, rules, blinding, scoring | After the pilots, the leak audit, the gradient cross-check and VJ's approval (Oct 25 authorisation in place of Vihaan's sign-offs), and before any study robot |

Each part's SHA-256 is recorded in `PHASE2_FREEZE.md` when it is frozen.

## 1. Preconditions

| Item | Status |
|---|---|
| Phase 1: 20 clean P0 v5b robots with full training records | Done Oct 22 |
| Layer 1 limits (success threshold 0.1187 m/s; combined check 2 limit) | Done Oct 22 |
| C1: the Oct 9 RSL-RL acceptance gate | Accepted by VJ in place of Vihaan, Oct 25, on the repository's record (gate captures replayed in CI on 5.0.1, 5.4.1, 5.5.1). The original gate report is still sought |
| Phase 0B Q1 (per-term formula) and Q4 (meaning) | Approved by VJ in place of Vihaan, Oct 25, through a documented review, with the assumptions and reading in section 6 |
| Independent gradient cross-check (phase 0B Q1 rule) | **Pending**, needed for part B |
| C3 kept, with no extra compute | Decided by VJ, Oct 22 |
| Arm pilots (part A, 16 robots) and amendment 1 pilots (4 robots) | Done Oct 24–25; 3 arms kept |
| Leak audit on the pilot robots' records (part A) | **Pending**, needed for part B |

## 2–3. Arms, pilots and leak audit (part A)

Frozen separately, in `PHASE2_PART_A.md` (Oct 23, 2026), and amended once (`PHASE2_PART_A_AMENDMENT1.md`, Oct 24).

### 3a. The study's arms (from the pilots)

| Label | Change from P0 v5b | Planted cause | Pilot evidence |
|---|---|---|---|
| Control | none (robots 101–105 from phase 1) | — | — |
| R-slip | `feet_slide` weight × 0.1 | reward | seeds 201–202 passed |
| Φ-slip-mild | training static friction 0.3–0.6 (dynamic 0.75 ×) | physics | seeds 219–220 passed |
| Φ-delay | no command delay in training (`max_delay` 0) | physics | seeds 221–222 passed |

**Not in the study, kept in the record:** Φ-slip (0.2–0.5 and 1.0–1.2), R-jitter (× 0.1 and × 0.03), Φ-jitter (judged
on jitter) and R-hop (× 10 and × 30). Their pilot robots and the reasons they were dropped are in
`evidence/phase2/pilots/README.md`.

**Φ-delay is a delay-fragility intervention.** It is the same training change as the dropped Φ-jitter arm, judged by a
different, predefined symptom (check 1's latency flag, amendment 1). It is not counted as a jitter cause.

**What a reduced design can and cannot show.** Two arms share a symptom with different causes (R-slip and
Φ-slip-mild), which is what C2 needs. The study cannot show anything about jitter or hopping.

### 3b. Intended-symptom patterns (manipulation check, reported only)

Amendment 1 asked for the symptoms to be revisited as patterns across several measures. They are used **only** to
report whether each study robot shows its arm's intended change. They never select robots, never enter a feature
table and never enter scoring, so they cannot be tuned to favour any approach.

| Arm | Pattern (all parts must hold) |
|---|---|
| R-slip | foot slip p95 > 0.1185 m/s, and nominal success ≥ 0.85 |
| Φ-slip-mild | foot slip p95 > 0.1185 m/s, and nominal success ≥ 0.85 |
| Φ-delay | check 1 scored and flags `action_latency_ms`, and foot slip p95 ≤ 0.1185 m/s |
| Control | none of the above |

Each arm reports the share of its robots that show its pattern, and every other measure that moved beyond the
controls' maximum (as in the pilot reports). The pilots showed that most interventions move several measures at once;
this table records that rather than assuming one symptom per cause.

## 4. Study robots (part B)

- **Seeds and recipe.** Each kept arm trains 5 robots on fresh seeds, fixed in advance, with the full training record
  (EvidenceRecorder, as phase 1). No pilot seed (201–222) is reused.

  | Arm | Assigned seeds |
  |---|---|
  | R-slip | 301–305 |
  | Φ-slip-mild | 306–310 |
  | Φ-delay | 311–315 |

  Seeds 316–345 are not used. There are no replacements (see "Failures are kept").
- **Every robot is fully measured, qualified or not:**
  - nominal PhysX over 256 episodes;
  - check 1 on the v2 envelope;
  - check 2 with the hop measures;
  - Newton.
- **Failures are kept.** Every assigned robot is in the primary set, whether or not it qualifies, so that poor
  performance cannot drop out through replacements.
  - **Qualifies** means nominal PhysX success of at least 0.85 over 256 episodes, where an episode succeeds if its mean
    velocity-tracking error is below 0.1187 m/s (the frozen v2 threshold, `frozen_success_p0v5b.json`).
  - **A run failure is not an outcome.** If training crashes or leaves no final checkpoint (an infrastructure fault,
    not a poor policy), that seed is retrained once from the start, and the fault is reported. If it fails again, the
    robot is missing for every approach and the report says so.
- **Analysis sets.**
  - **Primary:** all 15 assigned robots and the 5 controls, 20 in all.
  - **Secondary, conditional on qualification:** only robots that qualify, labelled as conditional on qualification.
  - Qualification rates are reported per arm as an outcome of the intervention.
- **Controls.** Controls 101–105 are the first five seeds assigned in phase 1, and all five qualified there (seeds
  106 and 112 did not and were replaced by 121 and 122, which are not controls). So controls were not selected on
  performance.
- **Controls, measurement.** The 5 control robots (101–105) were not used to develop any attribution rule; the rules in section 7
  are fixed without data. They are re-measured under phase 2's measurement code, so that every robot's behaviour
  features come from the same code version. A measurement is reused only where its code is unchanged, which is checked
  by version hash. Their phase 1 training records (TensorBoard and recorder) give B and C what they need.
- **Pilot robots are excluded.** No pilot robot (seeds 201–222) is a study robot, a control or a test robot, or enters
  any feature table, rule or score. They are used only for the leak audit (part A), which is reported.
- **GPU estimate.** The amendment 1 pilots took about 55 minutes per robot for training, nominal, check 1 and check 2;
  with Newton (check 5), about 1 hour.
  - 15 study robots: best case about 15 hours, worst case about 19 hours (slower runs).
  - Re-measuring the controls (check 1, check 2 and Newton under phase 2's code): about 2–3 hours.
  - Each retrained run failure: about 1 hour.
  - In all: best case about 17 hours, worst case about 23 hours.

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
- **The identity it rests on.** A = Σ_k T_k + V, where V is the GAE of zero reward with the trainer's critic. It holds
  only if every part uses the same episode masks (dones), γ, λ and bootstrap handling. RSL-RL's time-out bootstrap
  (adding γ·V at a time-out) belongs to V, not to any term. The recorder computes all parts from the same masks
  (`sim2gate.training.evidence`, `_advantage_parts`), and the exporter checks the identity on every update: the
  largest relative error of A − (Σ_k T_k + V) must be at most 10⁻⁵, or that update is flagged and left out.
- **Computation:** from the recorder's per-update covariance of [A, T_1…T_K, V-part], with the closed form verified
  numerically in phase 0B.
- **Normalisation:** correlation is unchanged by RSL-RL's per-rollout advantage normalisation (subtract the mean,
  divide by a positive scale), so ρ_k is computed on unnormalised values.
- **The formula and its assumptions (reviewed Oct 25).** Centre A and T_k, and regress T_k on A: T_k = β_k A + R_k
  with ⟨A, R_k⟩ = 0. Then A − T_k = (1 − β_k)A − R_k, and

  ρ_k = (1 − β_k)‖A‖ / √((1 − β_k)²‖A‖² + ‖R_k‖²),

  valid when both A and A − T_k have nonzero variance.
- **How to read it.**
  - ρ_k < 0 (β_k > 1) means the advantages without term k are **negatively correlated** with the trainer's. It does
    not by itself mean the PPO gradient or the optimizer update is reversed.
  - If R_k = 0, ρ_k is exactly +1 or −1, by the sign of 1 − β_k. These are valid values and are kept.
  - If R_k = 0 and β_k = 1, A − T_k is constant and ρ_k is undefined.
  - The fixed-critic counterfactual is a valid, defined comparison. It does **not** reproduce training without the
    term, which would also relearn the critic.
  - ρ_k says how much term k steers each update's advantages. It says nothing by itself about where training ends up
    (phase 0B Q4).
- **Status:** this is **an extension beyond V1.4**, which concerns bootstrapped reward shaping. Phase 0B **passed its
  recorded decision rules**, with recorded limits:
  - ρ beat the variance ratio Var(T_k)/Var(A) by only 0.002 in the pooled comparison;
  - within runs, ρ tracked the foot-slip term weakly (0.053);
  - the critic-refit yardstick was weak;
  - it measured gradient directions, not replayed Adam updates.

  ρ therefore enters phase 2 as **a candidate diagnostic whose added value remains to be tested**, which is what C3
  tests. No report claims that ρ reliably tracks ordinary reward terms.
- **Undefined and invalid values.** All moments are computed in float64. For each update and term:
  1. If any moment is not finite, ρ_k is invalid for that update.
  2. A is treated as constant, and ρ_k undefined, if Var(A) ≤ 10⁻¹⁰ × mean(A²) (centred variance negligible against
     the vector's own size) or mean(A²) = 0.
  3. A − T_k is treated as constant, and ρ_k undefined, if Var(A − T_k) ≤ 10⁻¹⁰ × max(mean((A − T_k)²), Var(A)).
  4. A computed |ρ_k| above 1 + 10⁻⁶ is invalid (a numerical fault, reported); values within that margin are clipped
     to [−1, 1].

  Undefined and invalid updates are left out of the window average and counted. If more than half of a window's
  updates are undefined or invalid, that feature is missing (section 7).
- **What C contains:** C is B plus ρ_k only. It does not include β_k, R_k, the variance ratio or any other phase 0B
  quantity. A comparison that adds those would be named separately.
- **ρ carries magnitude information that B's primary inputs exclude.** T_k is built from the *weighted* term, so ρ_k
  partly reflects how strongly a term is weighted. A C-over-B gain could therefore come from that magnitude rather
  than from ρ's specific form. Section 8 compares against B-var: B plus Var(T_k)/Var(A) per term per window. B-var
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

**Folds.** The unit is the independently trained robot. Fold *i* (1–5) holds each arm's *i*-th primary robot (by
seed order) and control robot 100 + *i*. That is 4 robots per fold, 20 in all. Episodes and log rows are never split
across folds. Fold membership depends on arm, so it is computed **only inside the sealed scoring program** from the
key (section 8). No fold assignment, seed order or arm-balanced ordering is ever exported or committed.

**Preprocessing (training folds only).**

1. Z-score each feature with the mean and standard deviation of the training folds.
2. A feature with zero standard deviation in the training folds is dropped for that fold.
3. A missing value is set to the training-fold mean (0 after z-scoring) and counted.
4. A feature missing for more than 20% of the **training-fold** robots is dropped for that fold, for every approach
   that uses it. The rule never looks at held-out robots. How often each feature was dropped is reported.

**Question 1, classification of the planted intervention.**

- **Rule:** nearest centroid, Euclidean distance, equal feature weights.
- **Labels:** Control, R-slip, Φ-slip-mild, Φ-delay.
- **Ties:** go to the first label in that order.
- **Score:** accuracy over all 20 robots of the primary set (chance is 25%).
- **Also reported:** the reward-vs-physics accuracy, and the accuracy on the two slip arms alone (R-slip against
  Φ-slip-mild), which is the part of the question the behaviour measures are least designed to answer.

**Question 2, early warning.**

- **Inputs:** window 1 features only.
- **Rule:** ridge regression, penalty 1.0, predicting the log of the final foot slip p95.
- **Score:** Spearman correlation between predicted and measured values over all 20 primary robots.
- **Why slip only.** Jitter and hopping are no longer study symptoms. Delay fragility cannot be predicted from window
  1: Φ-delay's training is identical to the control recipe until the delay ramp starts at round 500. The latency cliff
  is still predicted the same way and reported, not scored.
- **A is excluded:** it has no training-time data.

## 8. Blinding, labels and scoring (part B) ★

- **Who does what.** Key holder and scoring operator: **Vihaan** (confirmed by VJ, Oct 25; key check below). He holds
  the private key on his own computer and runs stage 1 there. VJ authorised the pending personal sign-offs on Oct 25
  (see "Review corrections"); the key-holder role is separate and is not replaced.
- **Random IDs.** The driver gives every study robot and every control a random ID from a cryptographic random source.
  IDs are not derived from seeds, names or order.
- **The key.** The key maps ID to arm and seed. It is encrypted to the key holder's public key, as in v1 (Vihaan's RSA
  3072-bit key of Oct 16, SHA-256 `9db064c8…`). **Key check, Oct 25:** the public key VJ supplied matched that hash;
  a random word sealed to it (RSA-OAEP, SHA-256) was opened by Vihaan on his own computer and returned correctly, so
  he still holds the private key. The plain key is not kept on the instance or in the repository.
- **Feature tables.** The exporter writes them under random IDs, with rows sorted by ID. They contain no fold, seed,
  arm, order or run name. Their SHA-256 hashes and the frozen scoring code are committed.
- **Sealed scoring, stage 1 (key holder, on his computer).** He runs the frozen scoring program with the key. The
  program builds the folds itself from the key (section 7). For each fold it uses only the four training folds'
  labels and never reads the test fold's labels. It writes predictions per robot ID for every approach, and nothing
  about folds. He sends the predictions file and its SHA-256.
- **Stage 2 (after the predictions hash is recorded).** Vihaan releases the key. Stage 2 checks the hashes and computes
  the scores.
- **Who stays blind.** VJ and Claude see no labels until the predictions are committed.

**Criteria (as the frozen plan).**

| Criterion | Passes if |
|---|---|
| C2 | B's or C's accuracy is at least 20 points above A's |
| C3 | C's accuracy is at least 10 points above B's; or C's early-warning Spearman is above B's, with the 95% bootstrap interval of the difference above zero |

**The bootstrap (C3's early-warning test).** A paired bootstrap over robots: each resample draws robots with
replacement, stratified by arm (controls as a fifth stratum), and keeps each drawn robot's B and C predictions together.
It resamples the **fixed out-of-fold predictions**; it does not refit the procedure. It therefore estimates the
uncertainty that comes from which robots were sampled, given the fitted models, not the variability of refitting.
10,000 resamples, seed 0; the interval is the 2.5th to 97.5th percentile of the Spearman difference C − B.
Resamples in which the Spearman correlation is undefined (all drawn values tied) are redrawn, and their number is
reported.

**Interpreting C3.** C3 is scored exactly as the plan says, C against B. The gain is attributed to **ρ specifically**
only if C also beats B-var under the same rule. If C beats B but not B-var, the report says: **"no demonstrated
advantage over the variance baseline."** It does not conclude that term magnitude caused the gain.

**Reported, not scored:**

- confusion matrices;
- per-symptom early warning;
- C-placebo and C against B-var;
- B_w and C_w;
- the secondary analysis set;
- question 1 without Φ-delay (Control, R-slip and Φ-slip-mild only);
- the intended-symptom table (section 3b).

**Arms dropped in the pilots.** Under part A as first written, dropping two or more arms meant C2 and C3 were not
tested, and that is reported. Amendment 1, approved before its pilots, replaced that rule with a final pilot round and
a hard stop that was not triggered. C2 and C3 are scored on the four labels above, and the report states:

- the reduced design;
- that the arms were chosen after seeing pilot results;
- every dropped arm.

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
- **D5 (revised).** Hopping is not a study symptom. Intended-symptom patterns (section 3b) are a reported manipulation
  check only and never select robots or enter scoring.
- **D9 (new, revision 4).** The arms are Control, R-slip, Φ-slip-mild and Φ-delay. Seeds are as in section 4. Pilot
  robots are excluded from all confirmatory scoring.
- **D10 (new, revision 4).** Early warning (question 2) is scored on foot slip only; the latency cliff is reported.
- **D6 ★** Sealed two-stage scoring. Vihaan runs stage 1 with training-fold labels only; VJ and Claude stay blind until
  the predictions are committed.
- **D7.** Every robot is fully measured and kept; failures are reported per arm; there is a secondary analysis on all
  robots.
- **D8 ★** ρ_k is defined as in section 6 and labelled an extension beyond V1.4; C differs from B only by ρ_k. A gain
  is credited to ρ only if C also beats B-var, which carries the same weighted-magnitude information.

## Review corrections (revision 4, Oct 25, 2026)

VJ authorised a documented review in place of Vihaan's pending personal sign-offs (phase 0B Q1 derivation, Q4
statement, and the ★ decisions D1, D4, D6, D8). The review approved the per-term formula with assumptions, accepted
phase 0B as "passed the recorded decision rules" with its limits, and asked for these corrections, all made:

| Review item | Where |
|---|---|
| Formula assumptions; β > 1 means negative advantage correlation, not a reversed update; R = 0 gives ±1; undefined when constant; fixed critic ≠ retraining; identical masks, discounting and time-out bootstrap | Section 6 |
| A numerical rule for undefined ρ, constant vectors and non-finite values (the old 10⁻¹² × Var(A) test could not fire) | Section 6 |
| Missing-feature rule on training folds only | Section 7, preprocessing step 4 |
| All originally assigned robots primary; qualification defined numerically; qualified-only analysis labelled conditional | Section 4 |
| Paired bootstrap of fixed out-of-fold predictions, and what it estimates | Section 8 |
| B-var wording: "no demonstrated advantage over the variance baseline" | Section 8 |
| Folds built inside the sealed program; no fold or order exported | Sections 7 and 8 |
| Authorisation recorded; key holder and scoring operator named | Section 8 and this section |

**Still required before the freeze, unchanged by the authorisation:**

1. An independent gradient cross-check on phase 0A's 256-environment sample (cosine ≥ 0.999999).
2. The exporter, tested, and the leak audit on the pilot robots' records.
3. The scoring program, tested, with its hash committed.
4. Tests passing on GitHub for the exact study commit.
5. The original Oct 9 gate report, if it can be found. Replayed ρ values in CI show compatibility of today's code
   with the gate's captures; they do not replace the report.
