# Phase 1, P0 v5 pilot: why it failed, and what we learned (Oct 21, 2026)

**Result: PILOT FAIL.** Neither robot qualified, so P0 v5 is not frozen. P0 v5b (the payload phased in) replaces it,
approved by VJ on Oct 21. This note records why, so the same mistake is not repeated.

**Outcome (same day): the P0 v5b pilot passed.** With only the payload's timing changed, the same two seeds went from
1.6% and 80.5% to **92.6% and 95.7%**, with no check 1 flag inside the v2 envelope. This supports the explanation
below, though two seeds cannot show how often the old recipe would have failed.

## What was run

- Recipe: P0 v5 (`Sim2Gate-Velocity-Flat-Go2-P0-v5`, code `914132f`). That is P0 v4 plus the full v2 envelope in
  training: a 0–8 kg payload on the base, ground friction 0.35–1.2, and motor strength 0.8–1.0. 2000 rounds.
- Two robots, seeds 1 and 2, measured at nominal on 256 episodes against v1's frozen success threshold (tracking
  error at most 0.1133 m/s).
- The rule: both robots must succeed in at least 85% of episodes. Check 1, check 2 and Newton run only after a robot
  qualifies, so neither robot got them.

## Results

| Robot | Nominal success | Falls | Notes |
|---|---|---|---|
| p0v5_s1 | **1.6%** (4 of 256) | 0 of 256 | Median tracking error 0.83 m/s against a 0.11 limit: it stood still |
| p0v5_s2 | **80.5%** (206 of 256) | 0 of 256 | Walks, but 4.5 points short of 85% |

From seed 1's training log, compared with P0 v4 at the end of training:

| Training signal | P0 v4 | P0 v5, seed 1 |
|---|---|---|
| Speed-tracking reward | 1.36 | **0.23** |
| Mean total reward | 29.66 | 9.46 |

Seed 2's curves are in the next section: it was stuck too, for most of the run.

## Why it failed (our best explanation, not proven)

1. **The payload was there from the first round.** In P0 v5 the payload was a "startup" event. Each of the 4096
   training robots got one random payload, 0 to 8 kg, at the very start and carried it for all 2000 rounds. Some
   robots were learning to walk with up to 8 kg on their back from round 0. The Go2 itself weighs about 15 kg.
2. **Standing still is a safe trap.** Early on, a robot that tries to walk under a heavy load falls or pays large
   torque and jerkiness penalties. A robot that stands still pays almost nothing. Both seeds settled into
   standing by about round 170. Seed 1 never left; seed 2 broke out late (see the curves below).
3. **We were inconsistent.** P0 v5 already phased the motor-weakness condition in gently, on the same schedule as the
   command delay: none before round 500, the full range by round 800. The delay ramp was itself v1's fix (C7) for the
   same kind of problem. The payload, the hardest new condition, was the one we did not phase in.

4. **Same recipe, different luck.** Seed 2 trained under exactly the same conditions as seed 1: the same 0–8 kg
   payload range from round 0, the same friction and motor-strength ranges, the same code. Only the random seed
   differed (the starting network, and which training robot got which payload). Both fell into the standing trap at
   the same time. Seed 2 happened to climb out late, around round 1000–1400, and was still improving when training
   ended; seed 1 did not. So the payload does not make walking impossible to learn; starting with it makes learning
   fragile, and how soon (or whether) a run gets out of the trap is down to chance.

Both robots were measured **without** a payload (the nominal test removes it). Seed 1 stands still even with no load
on its back, so it never learned to walk at all; this is not a robot that walks but cannot carry 8 kg.

What we did not test: we did not run P0 v5 with more rounds or with fewer conditions, so we cannot rule out other
contributions, such as friction down to 0.35 together with weak motors. With two seeds we also cannot say how often
P0 v5 falls into the trap, only that it can.

## What the training curves show (exploratory, added Oct 21)

Computed after the fact from both seeds' training curves and two P0 v4 runs (Oct 15) for reference
(`curves_analysis.py`, output in `curves_output.txt`; input fingerprint `a4993632…`). Two failures and two references
are examples, not evidence of a general rule.

| | P0 v4 (2 runs) | P0 v5 seed 1 | P0 v5 seed 2 |
|---|---|---|---|
| Speed-tracking reward, round 100 / 200 / 1000 / 2000 | 0.38 / 0.89 / 1.35 / 1.38 | 0.32 / 0.32 / 0.27 / 0.31 | 0.33 / 0.33 / 0.39 / 1.30 |
| Speed error first below 0.5 m/s | round 164–178 | never | round 1422 |
| Tracking below half of P0 v4's from | — | round 170 | round 171 |

1. **The two seeds were indistinguishable for about 1000 rounds.** Both stopped improving their forward-speed
   tracking at about round 100, while P0 v4 kept climbing. Seed 2 was not "partly walking" all along: it was stuck
   like seed 1, broke out late, and reached P0 v4's tracking only at round 1760. Its 80.5% is most likely a robot
   that ran out of rounds, which fits the protocol's next candidate fix (more rounds) as well.
2. **What "standing" looked like in the training numbers.** In rounds 50–100 the P0 v5 robots paid about twice
   P0 v4's torque penalty (carrying the payload). After that they kept still: their penalties for joint
   acceleration, jerky actions and body rocking were about half of P0 v4's, and they stopped falling sooner. They
   did learn to turn on the spot (turning reward the same as P0 v4's); they did not learn to move forward.
3. **The warning was visible after about 3 minutes of a 33-minute training run.** By round 170 both seeds' tracking
   reward was below half of P0 v4's at the same round. Even without a reference run, the pattern was clear: long
   episodes (few falls) while the speed error stayed at about 0.75 m/s, barely better than at the start of training
   (0.80).
4. **Payload is the cause, not the other conditions.** Before round 500, P0 v5 differed from P0 v4 only in the
   payload and the lowest friction (0.35 instead of 0.4): motor weakness and delay were still off. The trap formed by
   round 170. P0 v5b kept the same friction and passed (92.6% and 95.7%).
5. **Sim2Gate's existing training monitor did not see it.** Its signals (ρ, sign flips, zero tails, reward shares)
   looked normal and the same for both seeds. They were built for reward-shaping problems, not for this. A simple
   "is it learning the task at all" check would have caught it.
6. **An observation, not a conclusion:** after about round 500 both seeds' exploration noise rose again (P0 v4's
   kept falling). Seed 2's break-out happened during that rise; seed 1's noise rose too, without a break-out.

**For the product.** A "learns to survive but not to move" warning, raised at under 10% of training, is a concrete
candidate for Sim2Gate's training diagnostics. Here it would have saved about an hour of GPU time across the two
robots. It is one example: before claiming it, it needs a preregistered test on new runs, including runs that recover
and runs that are slow but fine, to measure false alarms.

## Lessons for us and for anyone training robots this way

- **Bring hard conditions in gradually.** Learn to walk first, then add load, weakness, delay and slippery ground.
  This is the second time this lesson has shown up (v1's delay ramp, now the payload). From now on, every new
  training condition gets the ramp unless there is a reason not to.
- **Zero falls is not a sign of health.** A robot that stands still never falls. Always check whether it does the task
  (here, the tracking error), not only whether it stays up. This is also a direct example of what Sim2Gate is for:
  a policy can look safe and still be useless.
- **One seed is not enough to judge a recipe.** The same recipe gave 1.6% and 80.5%. A fragile recipe can pass on
  one lucky seed, which is why the pilot needs both seeds to qualify and the 20 clean robots allow replacements.
- **"Startup" randomization is one draw per robot for the whole run.** It is not per episode. For a condition that
  should vary, use a per-episode ("reset") draw.
- **Watch the training curves early.** Both seeds' speed-tracking reward fell below half of P0 v4's by round 170,
  about 3 minutes into a 33-minute run. A driver could check this automatically and stop a stuck run early.
- **The pilot did its job.** Two robots (about 2 hours of GPU) found the problem before 20 robots (about 19 hours)
  were trained on a broken recipe.

## What changed: P0 v5b

The payload is now drawn again at every episode, uniform from 0 to an upper bound. The bound is 0 kg before round
500 and rises to 8 kg by round 800, the same schedule as the delay and motor strength. Evaluation (check 2's play
mode) uses the full 0–8 kg from the start. Everything else is as P0 v5. Same pilot rule, same seeds.

If the P0 v5b pilot also falls short, the frozen protocol's next revisions are, in order: more training rounds, then
a narrower range for the condition that fails. A narrower range would be reported as a change to the envelope.
