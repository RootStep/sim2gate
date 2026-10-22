# v2 phase 2: protocol (DRAFT for VJ's approval; Vihaan reviews the starred items; not frozen)

Phase 2 is the cause study (v2 plan, "Layer 2"). It asks whether looking inside the training run tells us **why** a
robot has a flaw better than looking at the finished robot's behaviour (criterion C2), and whether ρ adds to standard
training diagnostics (C3). This file is frozen, with its SHA-256 recorded, before any phase 2 study robot is trained.
The arm pilots in section 3 run under this draft and are reported, but their results cannot change sections 4–7.

## 1. Preconditions

| Item | Status |
|---|---|
| Phase 1: 20 clean P0 v5b robots with full training records | Done Oct 22 |
| Layer 1 limits (success threshold 0.1187 m/s; combined check 2 limit) | Done Oct 22 |
| C1: Vihaan confirms the Oct 9 RSL-RL acceptance gate | **Pending** |
| Phase 0B: Vihaan's sign-off on Q1 (the per-term ρ derivation) and Q4 | **Pending**. C is computed with the closed form verified in phase 0B unless he says otherwise |
| C3 kept, with no extra compute (VJ, Oct 22) | Decided |

## 2. Arms ★

All arms start from P0 v5b (2,000 rounds, everything as phase 1) and change one thing each.

**Proposed change to the plan (decision D1): every arm keeps the same set of reward terms.** In the plan, two reward
arms remove a term and R-hop adds one. A removed term is logged as zero, and its ρ is exactly 1. An added term appears
in only one arm. Any training-side signal would then name the arm outright, which is the leakage the plan forbids, and
C2 and C3 would be won trivially. So:

- reward arms **weaken** a term instead of removing it;
- the body-height term is present in **every** arm, at a small base weight;
- arms differ only in reward weights or physics.

| Arm | Change from P0 v5b | Cause | Expected symptom | Pilot ladder (section 3) |
|---|---|---|---|---|
| Control | None: phase 1 seeds 101–105 (decision D2) | none | none | — |
| R-slip | Foot-slip penalty (`feet_slide`) weight × k | Reward | Foot slip | k = 0.1, then 0.03 |
| Φ-slip | Low-friction training: static friction 0.2–0.5 (dynamic × 0.75); slip penalty kept | Physics | Foot slip, possibly | as stated; fallback: high-friction-only training, 1.0–1.2 |
| R-jitter | Action-rate penalty (`action_rate_l2`) weight × k | Reward | Jitter | k = 0.1, then 0.03 |
| Φ-jitter | No command delay in training (as v1's P2); motor strength and payload unchanged | Physics | Jitter, and fragility to delay | as stated |
| R-hop | Body-height reward weight raised from its base value | Reward | Hopping instead of walking | weight × 10, then × 30 of the base |

The body-height term rewards the base being higher than its nominal standing height. Its base weight is small enough
that the control recipe's behaviour is unchanged; the pilot confirms this on one robot.

## 3. Arm pilots (readiness gate "validate the experimental arms")

- **What is run:** each arm's first ladder step, on 2 pilot seeds (201–212). The control recipe with the base
  body-height weight runs on 1 seed.
- **Measured:** nominal PhysX success over 256 episodes, and check 2.
- **Symptom thresholds:**
  - slip: foot slip p95 above the clean maximum of 0.1124 m/s;
  - jitter: the jitter share above the clean maximum of 0.00935;
  - hop: flight share (section 4) above the control's maximum.
- **An arm passes its pilot** when both seeds show its symptom and both qualify (at least 85% success over 256
  episodes, v2 threshold).
- **If a step fails,** the next ladder step is tried on 2 new seeds. If every step fails, the arm is dropped and the
  report says so. Dropping one arm is not a reason to change another.
- **Control check:** the base body-height weight passes if the control pilot qualifies and its flight share is within
  the clean robots' range.
- **Cost:** about 45 minutes per pilot robot. One pass is about 11 robots, about 8 GPU hours; failed steps add more.
  This is not in the plan's 24 hours for phase 2 (decision D3).
- **Results:** the pilot results fix each arm's step. Pilot robots are development robots: never study robots, never
  test robots.

## 4. Study robots

- **Seeds:** each flawed arm trains 5 seeds, 301–325 assigned in order.
  - Training uses the full training record (EvidenceRecorder, defaults as phase 1).
  - Measurement is as in phase 1: nominal PhysX (256 episodes), check 1 on the v2 envelope, check 2, and Newton.
- **Qualification:** a study robot that does not qualify (85% over 256 episodes) is replaced by the next spare seed
  (326–335). It stays in the record. More than 2 replacements in one arm stops the run for a decision.
- **New measured quantity, flight share** (decision D5): the share of control steps in check 2's run with no foot in
  contact, taken from the same contact signal. It is reported for every robot. The 5 control robots are re-measured
  with check 2 so they have it. Their other check 2 values stay as recorded.
- **GPU estimate:** 25 robots × about 55 minutes is about 23 hours, as the plan says. Replacements and the control
  re-measure add about 1–3 hours.

## 5. Blinding and leakage ★

- **Random IDs.** The driver gives every study robot (including the 5 controls) a random ID. The ID-to-arm key is
  encrypted to Vihaan's public key, as in v1. The plain key never exists on the instance or in the repository after
  encryption.
- **Measured signals only.** Feature extraction reads only:
  - the robot's measurement files;
  - its TensorBoard event file (training curves);
  - the recorder's `updates.jsonl`.
- **Never read:** run folder names, task names, saved configs (`env.yaml`, `agent.yaml`), seeds, or reward weights.
  A test enforces this: the extractor is given files with the names stripped.
- **Same term set.** Every arm logs the same reward-term names (section 2), so the term list itself identifies no arm.
- **Sealing.** The feature tables are written under random IDs and hashed before the key is released.
- **Order of release:**
  1. The frozen analysis code and the feature-table hashes are committed.
  2. Vihaan releases the key.
  3. The analysis runs once.

## 6. Approaches and rules, fixed now ★

**Features.** Each approach gets one feature vector per robot. Every feature is z-scored using only the training fold.

| Approach | Features |
|---|---|
| A, behaviour only | Final measurements: nominal success (v2 threshold); check 2's four scored metrics, contact force and flight share; check 1 cliff index for each of the five conditions (0–9, 10 = no cliff); check 5 fall-free share and error ratio |
| B, standard training diagnostics | For each reward term, its per-step reward (TensorBoard `Episode_Reward/<term>`). Plus action std, value loss and learning rate from the training logs, and KL and clip fraction from the recorder. Each signal is averaged over three windows of training: rounds 0–399, 400–999 and 1000–1999 |
| C, B plus ρ | B's features plus, for each reward term, ρ_k (the phase 0B closed form, from the recorder's per-update moments), averaged over the same three windows |

**Question 1, attribution.**

- **Task:** name each robot's arm, one of the six labels.
- **Rule:** nearest centroid in the z-scored feature space, with Euclidean distance and equal weight per feature.
- **Validation:** leave one seed index out. For fold *i*, fit on the other 4 seeds of every arm and predict that arm's
  *i*-th robot. 30 robots, 30 predictions.
- **Score:** accuracy. The "reward vs physics" accuracy is also reported.

**Question 2, early warning.**

- **Task:** using only rounds 0–399 (the first 20%), predict each robot's final symptom size, separately for:
  - foot slip p95;
  - jitter share;
  - flight share.
- **Rule:** ridge regression with penalty 1.0 on the z-scored window-1 features of B, or of C, with the same folds.
- **Score:** Spearman rank correlation between predicted and measured values over all 30 robots, averaged over the three
  symptoms.
- **No question 2 for A:** A has no training-time data.

**Criteria (as the frozen plan).**

| Criterion | Passes if |
|---|---|
| C2 | B's or C's attribution accuracy is at least 20 points above A's |
| C3 | C's attribution accuracy is at least 10 points above B's; or C's early-warning Spearman is above B's, with the 95% bootstrap interval of the difference above zero |

The bootstrap uses 10,000 resamples of the 30 robots, stratified by arm, with seed 0.

**Reported, not scored:**

- confusion matrices;
- per-symptom early-warning results;
- results without the Φ-jitter arm. Its change only starts at round 500, so window 1 cannot see it.

**Inconclusive results.** If an arm is dropped in the pilots, the scores are computed on the remaining arms. The report
states the reduced design. If two or more flawed arms are dropped, C2 and C3 are reported as not tested.

## 7. What phase 2 delivers, and its limits

**Delivers:**

- C2 and C3 results under these rules;
- the confusion matrices;
- every robot's measurements and training record in one bundle with a SHA-256 manifest.

**Limits** (as in the plan):

- 5 seeds per arm show only large differences.
- The planted causes are cleaner than real training runs.
- The attribution rules are ours. A better rule for B could narrow C's lead.

The flawed robots also serve phase 3, fix verification, which has its own protocol.

## Decisions for VJ (★ = Vihaan reviews too)

- **D1 ★** Same reward-term set in every arm: reward arms weaken a term instead of removing it, and the body-height
  term is in every arm. This changes the plan's arm definitions to close a leakage path.
- **D2.** Control arm = phase 1 seeds 101–105, chosen by seed order and not by results.
- **D3.** Arm pilots as in section 3: 2 seeds per ladder step, symptom thresholds from Layer 1. About 8 extra GPU
  hours, more if steps fail.
- **D4 ★** Features, windows and fixed rules as in section 6: nearest centroid, ridge with penalty 1.0, leave one seed
  index out.
- **D5.** New flight-share metric for R-hop. The 5 controls are re-measured on check 2.
- **D6 ★** Blinding as in section 5. Vihaan holds the key; analysis runs once after the key is released.
- **D7.** Replacement rule: spare seeds 326–335; more than 2 replacements in one arm stops the run.
