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

Seed 2's training log has not been compared yet.

## Why it failed (our best explanation, not proven)

1. **The payload was there from the first round.** In P0 v5 the payload was a "startup" event. Each of the 4096
   training robots got one random payload, 0 to 8 kg, at the very start and carried it for all 2000 rounds. Some
   robots were learning to walk with up to 8 kg on their back from round 0. The Go2 itself weighs about 15 kg.
2. **Standing still is a safe trap.** Early on, a robot that tries to walk under a heavy load falls or pays large
   torque and jerkiness penalties. A robot that stands still pays almost nothing. Seed 1 settled into standing.
   Seed 2 mostly escaped the trap, but not fully.
3. **We were inconsistent.** P0 v5 already phased the motor-weakness condition in gently, on the same schedule as the
   command delay: none before round 500, the full range by round 800. The delay ramp was itself v1's fix (C7) for the
   same kind of problem. The payload, the hardest new condition, was the one we did not phase in.

4. **Same recipe, different luck.** Seed 2 trained under exactly the same conditions as seed 1: the same 0–8 kg
   payload range from round 0, the same friction and motor-strength ranges, the same code. Only the random seed
   differed (the starting network, and which training robot got which payload). One robot stood still and the other
   walked in 80% of episodes. So the payload does not make walking impossible to learn; starting with it makes
   learning fragile, and whether a given run falls into the standing trap is down to chance.

Both robots were measured **without** a payload (the nominal test removes it). Seed 1 stands still even with no load
on its back, so it never learned to walk at all; this is not a robot that walks but cannot carry 8 kg.

What we did not test: we did not run P0 v5 with more rounds or with fewer conditions, so we cannot rule out other
contributions, such as friction down to 0.35 together with weak motors. With two seeds we also cannot say how often
P0 v5 falls into the trap, only that it can.

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
- **Watch the training curves early.** Seed 1's speed-tracking reward was far below P0 v4's long before round 2000.
  Comparing it with the previous recipe at a few checkpoints (for example rounds 300, 800 and 1200) would have shown
  the problem an hour earlier. A future driver could check this automatically.
- **The pilot did its job.** Two robots (about 2 hours of GPU) found the problem before 20 robots (about 19 hours)
  were trained on a broken recipe.

## What changed: P0 v5b

The payload is now drawn again at every episode, uniform from 0 to an upper bound. The bound is 0 kg before round
500 and rises to 8 kg by round 800, the same schedule as the delay and motor strength. Evaluation (check 2's play
mode) uses the full 0–8 kg from the start. Everything else is as P0 v5. Same pilot rule, same seeds.

If the P0 v5b pilot also falls short, the frozen protocol's next revisions are, in order: more training rounds, then
a narrower range for the condition that fails. A narrower range would be reported as a change to the envelope.
