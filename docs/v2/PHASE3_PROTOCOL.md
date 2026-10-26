# v2 phase 3: fix verification (DRAFT, revision 1, Oct 26, 2026)

**Question.** When Sim2Gate flags a problem and recommends a training change, does retraining with that change give a
policy that is better than simply retraining the flawed recipe on new seeds, both on the flagged symptom and on
disturbances no check ever measured?

**Status.** Draft for VJ's review. Direction approved by VJ on Oct 26 (D1–D5, new training runs). The exact recipes
(section 2) and numerical rules (sections 5–6) are the open review items. Nothing is trained, and no held-out
condition is measured, before this file is frozen with its SHA-256 in `PHASE3_FREEZE.md`.

**Disclosed up front.** Both arms were chosen after seeing phase 2's results (they are the two planted physics causes
that phase 2's study robots showed reliably). Phase 3 therefore tests whether the recommended fixes work for these
flaws; it says nothing about how often Sim2Gate finds a flaw.

## 1. Terms

- **Action latency** is the delay between the policy issuing an action and the motors applying it. It is implemented
  by Sim2Gate's DC motor model as a whole number of 5 ms physics steps, both in training (where earlier documents
  call it "command delay") and in check 1 (where it is the `action_latency_ms` condition). This file says "action
  latency" throughout.
- **Policy run:** one training run of 2,000 rounds on one seed, and the policy it produces. Not a physical robot.
- **Pair:** the fix run and the control run of one arm that share a seed.

## 2. Recipes, side by side

All runs: Unitree Go2 flat ground, Isaac Lab 3.0 / RSL-RL PPO as in phases 1–2, 2,000 rounds, the same 11 reward
terms and order, motor strength 0.8–1.0 and payload 0–8 kg both ramped in over rounds 500–800, sensor noise at the
task default. Only the rows below differ.

| Setting | Baseline P0 v5b (phase 1) | Φ-delay **control** (flawed, = phase 2 arm) | Φ-delay **fix** | Φ-slip-mild **control** (flawed, = phase 2 arm) | Φ-slip-mild **fix** |
|---|---|---|---|---|---|
| `feet_slide` weight | −0.25 | −0.25 | −0.25 | −0.25 | **−0.50** (×2) |
| Other 10 reward weights | P0 v5b | P0 v5b | P0 v5b | P0 v5b | P0 v5b |
| Static friction (training) | 0.35–1.2 | 0.35–1.2 | 0.35–1.2 | **0.3–0.6** | **0.35–1.2** |
| Dynamic friction (training) | 0.2625–0.9 | 0.2625–0.9 | 0.2625–0.9 | 0.225–0.45 | 0.2625–0.9 |
| Action latency (training) | 0–30 ms (0–6 steps), ramped in over rounds 500–800 | **none (0 ms)** | **0–30 ms, ramped in over rounds 500–800** | 0–30 ms, ramped | 0–30 ms, ramped |

**Resolving the recipe question.**

- **Φ-delay fix = P0 v5b exactly.** Check 1's report for a latency flag says "randomize action latency over 0–30 ms
  in training, then retrain". Applied with the recipe's own latency mechanism (which ramps it in), that restores the
  baseline recipe.
- **Φ-slip-mild fix = P0 v5b with the slip penalty doubled.** Check 2's report for a slip flag says "add or raise the
  foot-slip penalty; widen friction randomization". Both parts are applied as a package (VJ, D2): friction is widened
  to 0.35–1.2, the range check 1's own fix text gives for the v2 envelope, and the penalty is raised ×2. The report
  gives no number for "raise"; ×2 is fixed here and piloted (section 4). The result tests the package; it cannot say
  which part helped.
- **R-slip** (restoring the slip penalty) would be a useful positive control but is not included. Adding it would cost
  6 more runs (about 7 GPU hours). Decision for VJ.

## 3. Runs and seeds

| Arm | Pairs (fix and control on the same seed) | Spares, used in order |
|---|---|---|
| Φ-delay | 411, 412, 413 | 421, 422, 423 |
| Φ-slip-mild | 414, 415, 416 | 424, 425, 426 |

12 policy runs in all; pilot seeds 401–404 (section 4) are never study runs. Each pair trains its fix and control
with the same seed, the same budget and the same code version, in the order fix then control.

- **Every run is measured fully,** qualified or not: nominal PhysX (256 episodes), check 1 on the v2 envelope, check 2,
  the held-out set (section 5) on PhysX and on Newton.
- **Not qualifying is an outcome.** A run below 0.85 nominal success stays in its pair and is judged like any other.
- **Run failure** (training crashes or leaves no final checkpoint): the run is retried once on the same seed. If it
  fails again, the whole pair moves to the arm's next spare seed (both runs retrained), and the failure is reported.
  Phase 2 showed a crash can repeat exactly on the same seed, which is why a second failure moves the pair.
- **Stop rule:** if an arm uses all three spares, the run stops for a decision.

## 4. Pilot (development only)

Only one recipe is new: the Φ-slip-mild fix with the slip penalty ×2. The others are P0 v5b and the two phase 2 arms,
which have already trained cleanly.

- **Runs:** the Φ-slip-mild fix on seeds 401 and 402.
- **Measured:** nominal PhysX (256 episodes), check 2, and the **development** disturbance set (section 5), never the
  held-out set.
- **Pass:** both runs qualify (nominal success ≥ 0.85) and both have slip p95 at or below 0.1185 m/s (phase 2's slip
  threshold).
- **If it fails:** one fallback candidate, the penalty ×1.5, on seeds 403 and 404, judged the same way. If that also
  fails, phase 3 runs with the Φ-delay arm only and the report says the slip fix was not tested.
- Candidate order, seeds and the pass rule are fixed here and do not change after a pilot result.

## 5. Disturbance sets

All five check 1 conditions vary **together** (friction, added payload, motor strength, action latency, sensor
noise), which no check measures. Each episode draws each condition independently and uniformly from its v2 check 1
grid values **up to the plausible edge** (the first 7 of the 10 values: severity 0 to 1), so every episode is a
realistic combination:

| Condition | Values drawn from |
|---|---|
| Ground friction (static) | 0.8 → 0.35 |
| Added payload | 0 → 8 kg |
| Motor strength | 1.0 → 0.8 |
| Action latency | 0 → 30 ms (whole 5 ms steps) |
| Sensor noise | 1× → 2× the task default |

| Set | Use | Episodes per run | Sampling seed |
|---|---|---|---|
| Development | Pilots only | 256 | 20261101 |
| **Held-out (phase 3)** | Scoring | 512 | 20261103 |
| Reserved for phase 4 (B7) | Not used in phase 3 | — | 20261104 |

The draws are generated by a new, versioned definition (`joint_v2`), and each set's SHA-256 is recorded at the
freeze. The phase 4 seed is reserved now so that phase 4 gets combinations no one has looked at.

**Held-out success** is the share of the 512 held-out episodes that succeed under the frozen v2 success threshold
(mean velocity-tracking error below 0.1187 m/s, no fall), on PhysX. The same on Newton is reported, not scored.

## 6. Scoring (criterion C4)

For each pair, the fix **beats** the control when both hold:

1. **Symptom.**
   - *Φ-delay:* the fix's check 1 latency cliff index is at least one grid step higher than the control's (one step is
     5 ms). "No cliff found" up to 45 ms counts as index 10. If both runs have no cliff, the symptom is not improved.
   - *Φ-slip-mild:* the fix's check 2 slip p95 is at least 20% lower than the control's, **and** at least 0.02 m/s
     lower.
2. **Held-out success:** the fix's held-out success is at least 5 percentage points higher than the control's.

**C4 passes** if, **in each arm**, at least 2 of the 3 pairs beat their control. Every pair is reported with its
values and effect sizes (differences, with Wilson intervals for success rates), whatever the result.

**Reported, not scored:** Newton held-out success; each condition's marginal success within the held-out set;
nominal success; check 1 and check 2 in full; the original Φ-delay and Φ-slip-mild robots of phase 2 measured on the
held-out set, as context.

**Limits.** Three pairs per arm is a small acceptance experiment: it can show a large, consistent effect, not a small
one. Simulation only.

## 7. GPU estimate

| Item | Best case | Worst case |
|---|---|---|
| Pilot (2 runs; 4 if the fallback is needed) | 2.5 h | 5 h |
| 12 study runs, trained and fully measured (about 1.1 h each) | 13 h | 17 h |
| Spares (each pair about 2.2 h) | 0 | 6.5 h |
| **Total** | **about 16 h** | **about 28 h** |

## 8. Decisions for VJ

- **E1.** Recipes as in section 2, including the slip penalty ×2 (fallback ×1.5) and friction 0.35–1.2.
- **E2.** Whether to add R-slip as a positive control (6 more runs).
- **E3.** The scoring thresholds in section 6: one latency step; slip 20% and 0.02 m/s; held-out +5 points; 2 of 3
  pairs per arm.
- **E4.** The disturbance sets and seeds in section 5, including reserving 20261104 for phase 4.
- **E5.** Run-failure, spare and stop rules in section 3.
