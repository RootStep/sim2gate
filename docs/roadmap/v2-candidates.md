# Sim2Gate version 2: candidate changes

Ideas found while building and validating the MVP. None of them changes the MVP: its rules are frozen for the blinded
validation. Each one needs its own evidence before it becomes a rule.

## Direction (agreed Oct 17, 2026)

Version 2's central direction is **detect → improve → explain**. Validation isn't a stage of its own: it's a gate
inside every stage. A finding needs episode counts and error bars before it sets a threshold. A diagnostic must pass
the known-answer suite before it scores. A fix must beat a matched control before the report recommends it. The next
big decision comes from the blinded results, not from adding features.

**Blinded results are in (Oct 17).** The MVP failed A1 (P4 jitter missed), A2 (P1 named on motor strength, not
friction/mass) and A5 (one planted robot at 78% nominal success), and passed A3, A4 and A6
(`docs/mvp-plan/evidence/step4/`). Items 13–16 come from that run. Version 2's first blinded run should retest those
failures on newly trained robots, with every rule fixed before training (item 16).

1. **Detect.** Analyse the joint sweep (item 1) on all 23 blinded policies, with error bars (item 12). Question: do
   combined conditions reveal failures that one-at-a-time testing misses? Compare with the existing checks on extra
   failures caught, false alarms, and separation of weak from clean policies. Only then propose a threshold. The data
   now exists (bundle `s2g-step4-2026-10-17.tgz`); because the key is released, this analysis is exploratory and any
   rule it suggests must be tested on new robots.
2. **Improve.** Config fingerprint (item 5), then test the recommended fixes (item 11) against matched controls on
   held-out conditions. The fixes the joint sweep suggests are mostly physics ones (e.g. widen mass and
   motor-strength randomisation). Testing them needs only check 1, the fingerprint and retraining, none of the
   research still in progress. Item 11 may become version 2's most valuable feature.
3. **Explain.** Known-answer suite (item 9), then reward-term influence in three levels (item 6), then trainer-aware
   check 6 (item 7). A reward term is only said to cause a behaviour with level-3 (retraining) evidence.

Alongside, as capacity allows: Newton repeatability (item 3), the Arena port (item 4), and a push pilot (item 10).
Research only: the prediction gap (item 8).

## 1. Joint sweep as a real check (found Oct 16, 2026)

**Finding.** The clean P0 v4 robot (seed 1) passes check 1: tested one condition at a time, it succeeds 91–98% of the
time across every realistic range. In the joint sweep, where all five conditions are drawn together, it succeeds in
only **74%** of the 504 episodes in which every condition is realistic at once. Some pairs are much worse together
than either alone (both in the harder half of their realistic range):

| Pair | Success together | Expected if independent |
|---|---|---|
| Extra base mass + weaker motors | 47% | 60% |
| Action latency + observation noise | 67% | 82% |
| Low friction + action latency | 74% | 83% |

**Why it matters.** One-condition-at-a-time testing can make a policy look safer than it is: on a real robot,
conditions stack. This is a differentiator worth building on.

**What version 2 would need.**
- A flag rule fixed in advance, e.g. "success with every condition realistic at once below X%", with X calibrated
  on clean baselines the way check 2's thresholds are.
- Evidence on more policies that the joint result separates weak from robust policies better than the
  one-at-a-time sweep does (the MVP's blinded run recorded the joint sweep for all 23 policies, report-only).
- Arena's simulation-based sensitivity analysis on the saved per-episode tables, for interactions beyond pairs.
- Episode counts and intervals for each pair cell (item 12). Each cell is a subset of the 504 episodes, so the
  smaller gaps (74% vs 83%) may be within sampling noise. Seed 1 alone can't set a general threshold.

Source: `mvp_runs/p5_pilot/joint_smoke.json` on `sim2gate-arena-2` (bundle `s2g-p5pilot-2026-10-16.tgz`).

## 2. Check 2 thresholds depend on the training recipe

The P0 v4 robots (trained with friction and delay randomization) are much smoother than P0 v2. The actuator-limit and
jitter thresholds calibrated on them would flag P0 v2 on those alone (`calibration/p0v4/observations.md`, item 3).
Version 2 could calibrate thresholds per training-recipe family, or report check 2 relative to the policy's own
recipe.

## 3. Newton run-to-run variation

PhysX results reproduce exactly; Newton runs vary slightly (one episode in 64 can flip between falling and not, error
ratio ±0.02; `evidence/pre-freeze/README.md`). Version 2 could run check 5's Newton side more than once, or with
more episodes, and report the spread.

## 4. Port to Isaac Lab-Arena

Planned since C2: checks 1, 2 and 5 as Arena variation and metric plug-ins, after the MVP validation.

---

Items 5–9 come from reading Chapters 21–22 of Capobianco, *Grace: The Value of Choice* (2026), on Oct 17, 2026.
Chapter 22 page numbers are from the e-book screenshots. The Chapter 21 screenshots show no page numbers, so those
items cite section, figure or exercise numbers instead. The book is cited, not copied: keep transcriptions of it out
of the repo.

## 5. Training-config fingerprint (p. 476)

**Finding.** The book's notes say PPO's results depend heavily on settings no equation shows: advantage normalisation,
value-loss clipping, learning-rate annealing and the number of epochs. Item 2 is already an example: the check 2
thresholds depend on the training recipe. The libraries also differ by default: RSL-RL normalises advantages once
per rollout, while SB3 and CleanRL normalise per minibatch.

**What version 2 would add.** For every policy, a report section recording trainer and version, advantage
normalisation mode, value clipping, learning-rate schedule (adaptive/fixed, target KL), epochs, minibatches, clip
range, entropy coefficient, γ, λ, seed, and the domain-randomisation ranges. Isaac Lab's training script already
saves `params/env.yaml` and `params/agent.yaml` in each run folder, so this mostly means parsing them.

**Uses.** It groups policies into recipe families for item 2, explains differences between policies, and flags
comparisons across different γ. Changing γ changes the objective being solved, while changing λ only changes the
estimator (Ex. 21.3). The randomisation ranges also let the report connect a joint-sweep weakness to the recipe, e.g.
"weak when mass and motor strength change together; training randomised neither". That should be stated as a lead to
test (item 11), not a proven cause. It is also the natural fix for item 14.

**Evidence needed.** None to record it. To use recipe families in a rule, show that thresholds calibrated per family
reduce false flags on the 23 blinded policies.

## 6. Check 3 should measure influence on the update, not share of return (Ch. 21, §21.2–21.3)

**Finding.** The check 3 example in the plan ("70% of return comes from a shaping term") measures each reward term's
share of the return. The book shows why that misleads. Any amount subtracted that doesn't depend on the action
leaves the expected gradient unchanged (Thm 21.1). In its Figure 21.1, offsetting every reward by 100 multiplies the
gradient's spread about a thousandfold, from SD 3.2 to 4264, without changing what the policy is pushed toward. So a
term can dominate the return and do nothing, or be small and steer the policy.

Caution: in locomotion, an alive bonus is not harmless in this way, because episode length depends on the policy.
That is exactly why share of return can't settle the question either way.

**What version 2 would add.** Attribute each reward term by how much it changes the centred advantages and the update
direction, using Vihaan's V1.4 estimator per term. Keep three levels of claim apart:

1. **Advantage change with the critic held fixed.** Cheap, and possibly computable post-hoc from the checkpoint
   (actor and critic) plus fresh rollouts with Isaac Lab's per-term rewards. But that critic learned the full reward,
   including the removed term, so its baseline still carries that term. V1.4's raw-critic versus shaped-critic fork
   is the starting point; Vihaan to confirm how it applies per term.
2. **Actor-gradient change** on the same minibatches.
3. **Behaviour change after retraining** without the term, with item 11's matched controls.

Levels 1–2 can support a diagnostic. Only level 3 supports a claim that removing or rebalancing a term changes what
the robot does.

**Evidence needed.**
- Negative control: a constant offset in a setting where it really is action-independent, i.e. fixed horizon with
  early termination switched off (or a toy task). Standard Go2 locomotion terminates on falls, so a constant there is
  not a valid negative control.
- Positive control: a planted shaping term that causes a known behaviour should score high.
- On policies with known shaping changes, the level-1/2 score should predict level-3 behaviour change better than
  return share does.

## 7. Check 6 flags must be trainer-aware (§22.6, Prop 22.3, Ex. 22.4, pp. 467–469, 474)

**Finding.** PPO's clip removes the incentive to move a sample further, not the possibility. Other samples in the
minibatch can push it further, and PPO has no monotone-improvement guarantee. Practitioners therefore monitor
measured KL and stop epochs when it passes a target: in the book's words, the trust region "reintroduced as a
diagnostic". Trainers apply that diagnostic differently. SB3's `target_kl` stops epochs, so the signal sits in epochs
completed. RSL-RL's default adaptive schedule lowers the learning rate when KL rises, so KL rarely spikes there and
the signal sits in the learning-rate trace. A generic "KL spike" flag, as in LLM tools such as rlwatch, would miss
RSL-RL problems.

**What version 2 would add.**
- KL read together with the learning-rate trace (RSL-RL) or epochs completed (SB3).
- Clip fraction split by the advantage's sign. The clip is asymmetric (p. 468): only the side the advantage favours
  loses its gradient.
- The existing action-std collapse flag, now with a reason: as a policy becomes nearly deterministic, the same
  parameter step changes behaviour much more (p. 472, Fisher eigenvalue). Report KL beside it.

**Evidence needed.** Each flag must fire on training runs with a planted cause (item 9) and stay quiet on clean
baselines before it counts toward the score.

## 8. Prediction gap: research, not a feature (Ex. 22.1, Fig. 22.1, pp. 471–472)

**Finding.** The surrogate PPO optimises agrees with the true objective only for small steps. In Figure 22.1 it
promises 10.3 where the truth is 5.9. And "the surrogate never announces that it has stopped being accurate".

**Idea.** Track predicted improvement against improvement measured on a fixed evaluation set during training.
Hypothesis: policies whose training showed large gaps are more fragile in check 1.

**Why it's only research.** It needs evaluation during training or closely spaced checkpoints. RSL-RL's default
checkpoint spacing is far beyond the range where the surrogate means anything. Belongs with Experiment T, not the
product roadmap, until it shows signal.

## 9. Known-answer suite for checks 3 and 6 (Ch. 21 trial, §21.6–21.7)

**Finding.** The book validates its estimators against exact answers on small problems and with planted errors.
Checks 1, 2 and 5 are validated by planted policies. Checks 3 and 6 need the same, with planted training bugs whose
effect is known.

**What version 2 would add.** Short Go2 (or toy) training runs with one planted cause each:
- time-limit truncation treated as termination (no bootstrap)
- a constant reward offset, under item 6's conditions (negative control)
- an action-dependent baseline (Ex. 21.1: biases the gradient)
- the advantage normalisation mode switched
- a lagging critic
- entropy coefficient set to zero (std collapse)

This also seeds the open-source "library of reproducible failure cases".

---

Items 10–12 came from reviewing a ChatGPT V2 proposal on Oct 17, 2026. Most of that proposal re-described checks 1,
2 and 5, which already exist; these were the new parts.

## 10. External pushes (deferred; pilot first)

Check 1 varies friction, mass, motor strength, latency and sensor noise, but applies no disturbances. Pushes are the
standard locomotion robustness test and Isaac Lab has a built-in push event.

**Status: deferred until the five-condition joint sweep has been evaluated.** Pushes test recovery, which none of
the five conditions exercises, so the joint-sweep results can't show whether pushes would add anything. When it comes
up, run a cheap pilot on the clean baselines. Add push strength as a condition only if it reveals reproducible
weaknesses the other conditions miss.

## 11. Verify the fix, not just the diagnosis

Today the report's "Fix" line is advice. Version 2 would test it: retrain with the recommended change and show the
policy improves on held-out conditions.

**Design.** For each recommended fix (e.g. widen friction randomisation), compare three arms:
1. the original policy;
2. retrained with the recommended change;
3. retrained with the **original recipe**: same training budget, different seeds (the matched control).

At least three seeds for each retrained arm, matching the P0 setup. Compare the spread of results across seeds, not
one run against one run. Without arm 3, an improvement could come from retraining randomness rather than the
recommendation.

**Held-out conditions.** Evaluate on conditions not used to choose the fix or calibrate any threshold. Fix the
held-out set and the success criterion before retraining.

**Milestone:** *given a frozen Go2 PPO policy, Sim2Gate finds its main physics and observation weaknesses,
reproduces the failures, and recommends changes that measurably improve held-out robustness over a matched retraining
control.* This needs no hardware, and it makes the report's recommendations a tested claim. It may become version
2's most valuable feature.

## 12. Error bars on every success rate

At 64 episodes, a 90% success rate carries a 95% interval of about ±7 points, and 50% about ±12. Some check 1
differences and joint-sweep pair gaps may sit inside that. Report Wilson intervals with episode counts, and set
thresholds with them in view. Do this before item 1 becomes a scored rule. The MVP's A5 failure is a live example: a
planted robot passed its 64-episode walking check at ≥ 80% and then measured 78% (50/64) in check 1. Pass/fail bars
near a threshold need more episodes (e.g. 256) or a rule that accounts for the interval.

---

Items 13–16 come from the MVP's blinded run and scoring (Oct 17, 2026; `docs/mvp-plan/evidence/step4/`). That run's
key is released, so its data can guide version 2 but cannot test it: every rule below is tested on newly trained
robots.

## 13. Quiet blinded runs

Found during step 4 (`docs/mvp-plan/FREEZE.md`): `blinded run` streams Isaac Lab's start-up output, which lists each
task's reward terms next to the blind ID. Send its subprocess output to files only, as `testset` does, and keep
those logs out of the runner's view until the verdicts are sealed.

## 14. Check 2's jitter threshold missed a mild jitter flaw (A1)

**Finding.** All three P4 robots (no action-rate penalty) had a higher jitter score (0.0122–0.0136) than all eight
clean P0 robots (0.0095–0.0111), but stayed under the frozen threshold 0.0162, set from three calibration seeds. P2
robots (no latency training) also scored 0.0133–0.0157. The ordering is right; the threshold is loose.

**What version 2 would do.** Calibrate check 2's thresholds on more clean seeds (all eight P0 v4 robots now exist),
with a rule written before new test robots are trained, e.g. clean maximum plus a fixed fraction of the clean spread,
or a recipe-family threshold (item 2). The gap between the jumpiest clean robot and the smoothest P4 was small
(0.0111 vs 0.0122), so a tighter threshold risks false alarms; the retest must include new clean robots.

## 15. Check 1 named the wrong parameter for narrow randomization (A2)

**Finding.** All three P1 robots (friction and base-mass randomization narrowed) got no-go, but two were flagged only
for weak motors (cliff at 0.85, exactly the threshold), not friction or mass. Narrow mass training showing up as
motor-strength sensitivity is physically plausible (both change the load on the legs), but it was not the planted
cause, and one P1 seed landed exactly on the threshold.

**What version 2 would do.** Report correlated parameters together and connect a flagged weakness to the training
ranges via the config fingerprint (item 5): "fragile to weaker motors; training never varied mass or motor strength".
Decide before the retest whether a load-related flag (mass or motor strength) counts as detecting P1, and say why.

## 16. Version 2's first blinded retest

Retest the MVP's failures on new robots, with C10-style rules fixed and approved before training: items 12 (more
nominal episodes), 13 (quiet runs), 14 (jitter threshold rule) and 15 (P1 rule). New robots: at least 5 clean P0
(false alarms under the new threshold), 3 P1 and 3 P4; P2, P3 and P5 results stand unless a change affects them.
Roughly 12–13 GPU hours. Vihaan holds the key again. Report honestly as a retest: the MVP failed first; the retest
uses rules fixed in advance on fresh robots.

## Considered and not added

Conservative policy iteration, natural gradient and TRPO computations (§22.2–22.5) are background, not features. The
book says the CPI bound is vacuous at practical step sizes, so it can't set thresholds. Points that belong to Vihaan's
paper, not the product: the ρ notation clash (the book uses ρ_t for PPO's likelihood ratio, p. 467), citing the
open question of how GAE bias interacts with the trust region (p. 475), and confirming the V1.4 identity under
RSL-RL's per-rollout normalisation.

Whether ρ predicts robustness belongs to Experiment T, not this roadmap. A useful design for it: does adding ρ predict
held-out check 1 failures better than nominal success, KL and clip fraction alone? Fix the comparison in advance, use
one or two features, test on held-out policies, and use policies that differ in how they were trained. If ρ adds
nothing, it stays a training diagnostic and isn't marketed as a transfer-risk predictor.

Not for v2: GRPO, general VLA evaluation, a custom simulator.

**Report wording (applies to all checks).** The book puts it as "I have exchanged a guarantee for a heuristic … and I
should say which one I am relying on when something goes wrong" (p. 471). Each finding in the report should say
which kind of evidence it rests on: measured in simulation (check 1 sweeps are controlled interventions, so they can
show a cause in simulation), a heuristic warning, or validated on hardware. A real-world cause needs hardware.
