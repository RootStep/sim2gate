# v2 phase 3: fix verification (DRAFT, revision 2, Oct 26, 2026)

**Question.** When Sim2Gate flags a problem and recommends a training change, does retraining with that change give a
policy that is better than simply retraining the flawed recipe on new seeds, both on the flagged symptom and on
previously unmeasured combinations of disturbances?

**Status.** Draft for freezing. Direction approved by VJ on Oct 26 (D1–D5, new training runs); revision 1's review
items E1–E5 approved the same day with amendments, all made here (listed at the end). Nothing is trained, and no
evaluation set is measured, before this file is frozen with its SHA-256 in `PHASE3_FREEZE.md`.

**Disclosed up front.**

- Both arms were chosen after seeing phase 2's results: they are the two planted physics causes that phase 2's study
  runs showed reliably. Phase 3 tests whether the recommended fixes work for these flaws; it says nothing about how
  often Sim2Gate finds a flaw.
- Each fix is tested as the package Sim2Gate's report recommends. A pass would show that the package helps; it would
  not show that it is the unique or best fix, nor which part of a package helped.
- The thresholds in section 6 are RootStep's internal acceptance requirements, not established industry standards.

## 1. Terms

- **Action latency** is the delay between the policy issuing an action and the motors applying it. Sim2Gate's DC motor
  model implements it as a whole number of 5 ms physics steps, both in training (where earlier documents call it
  "command delay") and in check 1 (its `action_latency_ms` condition). This file says "action latency" throughout.
- **Policy run:** one training run of 2,000 rounds on one seed, and the policy it produces. Not a physical robot.
- **Pair:** the fix run and the control run of one arm that share a seed.

## 2. Recipes, side by side

All runs: Unitree Go2 flat ground, Isaac Lab 3.0 / RSL-RL PPO as in phases 1–2, 2,000 rounds, the same 11 reward
terms and order, motor strength 0.8–1.0 and payload 0–8 kg both ramped in over rounds 500–800, sensor noise at the
task default. Only the rows below differ.

| Setting | Baseline P0 v5b (phase 1) | Φ-delay **control** (flawed, = phase 2 arm) | Φ-delay **fix** | Φ-slip-mild **control** (flawed, = phase 2 arm) | Φ-slip-mild **fix** |
|---|---|---|---|---|---|
| `feet_slide` weight | −0.25 | −0.25 | −0.25 | −0.25 | **−0.50** (×2; fallback ×1.5 = −0.375) |
| Other 10 reward weights | P0 v5b | P0 v5b | P0 v5b | P0 v5b | P0 v5b |
| Static friction (training) | 0.35–1.2 | 0.35–1.2 | 0.35–1.2 | **0.3–0.6** | **0.35–1.2** |
| Dynamic friction (training) | 0.2625–0.9 | 0.2625–0.9 | 0.2625–0.9 | 0.225–0.45 | 0.2625–0.9 |
| Action latency (training) | 0–30 ms (0–6 steps), ramped in over rounds 500–800 | **none (0 ms)** | **0–30 ms, ramped in over rounds 500–800** | 0–30 ms, ramped | 0–30 ms, ramped |

- **Φ-delay fix = P0 v5b exactly.** Check 1's report for a latency flag says "randomize action latency over 0–30 ms
  in training, then retrain". Applied with the recipe's own latency mechanism, which ramps it in, that restores the
  baseline recipe.
- **Φ-slip-mild fix = P0 v5b with the slip penalty raised.** Check 2's report for a slip flag says "add or raise the
  foot-slip penalty; widen friction randomization". Both parts are applied (VJ, D2): friction widened to 0.35–1.2, the
  range check 1's own fix text gives for the v2 envelope, and the penalty raised ×2, or ×1.5 if the pilot selects the
  fallback (section 4). The report gives no number for "raise"; these are fixed here.
- **No R-slip positive control** (VJ, E2).

## 3. Runs and seeds

| Arm | Pairs (fix and control on the same seed) | Spare seeds, used in order |
|---|---|---|
| Φ-delay | 411, 412, 413 | 421, 422 |
| Φ-slip-mild | 414, 415, 416 | 424, 425 |

12 policy runs. Pilot seeds 401–404 are never study runs. Each pair trains its fix and its control with the same seed,
the same 2,000-round budget and the same code version, fix first.

- **Every run is measured fully,** qualified or not: nominal PhysX (256 episodes), check 1 on the v2 envelope, check 2,
  and the phase 3 evaluation set (section 5) on PhysX and on Newton.
- **Not qualifying is an outcome.** A run below 0.85 nominal success stays in its pair and is judged like any other.
- **Run failure** (training crashes or leaves no final checkpoint): the run is retried once on the same seed. If it
  fails again, the pair moves to the arm's next spare seed and both of its runs are trained there. Phase 2 showed a
  crash can repeat exactly on the same seed, which is why a second failure moves the pair.
- **Every attempt is kept and reported,** including failed attempts and the abandoned half of a moved pair.
- **Stop rule.** An arm may move a pair at most twice (its two spare seeds). If a third move would be needed, the run
  stops **before** training a third spare pair, for a decision. The arm's result is then reported as incomplete.

## 4. Pilot (development only)

Only one recipe is new: the Φ-slip-mild fix with the raised slip penalty. The others are P0 v5b and the two phase 2
arms, which have already trained cleanly.

- **Candidate 1:** penalty ×2 (−0.50), seeds 401 and 402.
- **Pass:** both runs qualify (nominal success ≥ 0.85 over 256 episodes) and both have check 2 slip p95 at or below
  0.1185 m/s (phase 2's slip threshold).
- **Candidate 2, only if candidate 1 fails:** ×1.5 (−0.375), seeds 403 and 404, judged the same way. No further
  candidates.
- **Measured:** nominal PhysX, check 2, and the development set (section 5). Development-set results are
  **descriptive only**; no rule uses them. The phase 3 evaluation set is never run in the pilot.
- **Recorded before study training:** which candidate passed, in `PHASE3_FREEZE.md`.
- **If both fail:** the slip arm is not trained. The report says **"slip-fix verification not tested; two-arm C4 not
  established"**, and the Φ-delay arm is run and reported on its own.

## 5. Evaluation sets: previously unmeasured joint combinations

All five check 1 conditions vary **together**. Check 1 varies one condition at a time, so these are **previously
unmeasured joint combinations within familiar parameter ranges**: each value is one the checks already use, but the
combinations have never been measured. Individually plausible settings do not make every combination realistic;
results are read as robustness to compound stress, not as a deployment forecast.

**Values.** Each condition takes one of its v2 check 1 grid values up to the plausible edge (severity 0 to 1):

| Condition | The 7 values |
|---|---|
| Ground friction (static; dynamic = 0.75 × static) | 0.8, 0.725, 0.65, 0.575, 0.5, 0.425, 0.35 |
| Added payload (kg) | 0, 1.333, 2.667, 4.0, 5.333, 6.667, 8.0 |
| Motor strength (× nominal torque) | 1.0, 0.967, 0.933, 0.9, 0.867, 0.833, 0.8 |
| Action latency (ms) | 0, 5, 10, 15, 20, 25, 30 |
| Sensor noise (× task default) | 1.0, 1.167, 1.333, 1.5, 1.667, 1.833, 2.0 |

(Exact values: `sim2gate.checks.robustness.values(<condition>, "v2")[:7]`.)

**Disjoint sets, by construction.** There are 7⁵ = 16,807 combinations. The 31 that differ from nominal in at most one
condition are removed, because check 1 measures those. The remaining 16,776 are shuffled once with seed 20261103 and
cut into consecutive, non-overlapping blocks:

| Set | Use | Combinations |
|---|---|---|
| Development | Pilot only, descriptive | the first 256 |
| **Phase 3 evaluation** | Scoring | the next 512 |
| Reserved for phase 4 (B7) | Not opened in phase 3 | the next 512 |

The generator (`joint_v2`) checks that the blocks share no combination and records each block's SHA-256 at the
freeze.

**Identical conditions within a pair.** Fix and control are evaluated on exactly the same episodes: episode *i* gets
the same combination, the same velocity commands (Sim2Gate's fixed Go2 command set), the same initial-state
randomization (the same simulator seed) and the same 20 s length, for both runs of every pair. The evaluation
manifest (combinations, commands, simulator seed, episode length) is hashed and the hash is recorded with each
result.

**Evaluation success** is the share of the 512 episodes that succeed under the frozen v2 rule (mean velocity-tracking
error below 0.1187 m/s, no fall), on PhysX. Newton is reported, not scored.

**Phase 2's original Φ-delay and Φ-slip-mild runs** are measured on the evaluation set only after this file is frozen,
and are reported as context. They never change a rule or a result.

## 6. Scoring (criterion C4)

For each pair, the fix **beats** the control when both hold:

1. **Symptom.**
   - *Φ-delay: latency cliff, in ms.* Check 1 tests 0, 5, 10, …, 45 ms; a run's cliff is the first latency whose
     success falls below 0.8. A run with no cliff is **censored at above 45 ms**. The fix beats the control when its
     cliff is at least 5 ms higher. A censored fix beats any control whose cliff is at most 40 ms. A censored control
     cannot be beaten (its true cliff is unknown), so the symptom is then counted as not improved.
   - *Φ-slip-mild: check 2 slip p95.* The fix's value is at least 20% lower than the control's **and** at least
     0.02 m/s lower.
2. **Evaluation success:** the fix's is at least 5 percentage points higher than the control's.

**C4 passes** if, in each arm, at least 2 of the 3 pairs beat their control.

**Uncertainty, reported for every pair:** a Wilson 95% interval for each run's success rate; for the fix-minus-control
difference, a paired bootstrap over the 512 matched episodes (10,000 resamples, seed 0) and its 95% interval. These
are reported; C4's point thresholds above are unchanged.

**Incomplete outcomes.**

| Situation | Reported as |
|---|---|
| Both slip-fix pilot candidates fail | "Slip-fix verification not tested; two-arm C4 not established." Φ-delay reported on its own (2 of 3 pairs rule). |
| An arm stops under the stop rule | "That arm incomplete; two-arm C4 not established." Every attempt reported. |

**Reported, not scored:** Newton evaluation success; marginal success by condition value within the evaluation set;
nominal success; check 1 and check 2 in full; development-set results from the pilot; phase 2's original runs on the
evaluation set.

**Limits.** Three pairs per arm is a small acceptance experiment: it can show a large, consistent effect, not a small
one. Simulation only.

## 7. GPU estimate

Each policy run, trained and fully measured, takes about 1.1 hours.

| Item | Best case | Worst case |
|---|---|---|
| Pilot (2 runs; 4 with the fallback) | 2.5 h | 5 h |
| 12 study runs | 13 h | 17 h |
| Moved pairs (up to 2 per arm; each costs up to 4.4 h: two lost attempts, a lost partner run, and a new pair) | 0 | 17.6 h |
| **Total** | **about 16 h** | **about 40 h** |

## Amendments in revision 2 (VJ's review of revision 1, Oct 26)

| Item | Change |
|---|---|
| E1 | Approved. The passing pilot candidate is recorded before study training. |
| E2 | R-slip not added. |
| E3 | Approved as RootStep's internal requirements; latency scored in ms with the censored "no cliff" case defined. |
| E4 | Every value listed; pairs evaluated on identical episodes with hashed manifests; sets disjoint by construction, with the one-condition combinations check 1 measures removed; described as previously unmeasured joint combinations within familiar ranges. |
| E5 | Every attempt kept; the stop point is before a third moved pair; the budget recalculated. |
| Also | Paired bootstrap for differences; incomplete outcomes worded in advance; development results descriptive only; phase 2's runs measured only after the freeze. |
