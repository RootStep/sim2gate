# P0 v4 calibration: observations to watch (Oct 15, 2026)

From the check 1 and check 5 scores of P0 v4 seeds 1–3 (`checks15 --score-only`, Sim2Gate 4cf7ecf; runs measured
at 86f2c7d). Neither check flagged any seed. These are observations, not rules: they change no threshold or
decision, and the blinded run is scored with the rules as frozen. They are recorded so they are not lost, and so
the blinded results can be read against them afterwards.

## 1. Check 1: most cliffs sit one or two grid steps past the realistic edge

Each sweep has 10 values; value 6 is the realistic edge, and the step is 1/6 of the distance from nominal to the
edge. A cliff (the first value where fewer than 80% of episodes succeed) inside the realistic range would be
flagged; none is.

| Parameter (realistic edge) | Step | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| ground friction (0.4) | 0.067 | 0.267 (+2) | 0.267 (+2) | 0.333 (+1) |
| added base mass (3.0 kg) | 0.5 kg | none | none | 3.5 (+1) |
| motor strength (0.85) | 0.025 | 0.80 (+2) | none | 0.825 (+1) |
| action latency (30 ms) | 5 ms | 45 (+3) | none | 40 (+2) |
| observation noise (2.0x) | 0.167 | none | none | 2.33 (+1) |

(+n = n grid steps past the edge; "none" = no cliff within the sweep.)

Seed 3 fails at the first value past the edge on four of the five parameters: a clean, well-trained policy can
sit right at the boundary. Expect the same in the blinded set: a clean policy with cliffs at +1 is not a false
negative of check 1, and a policy flagged by a single grid step may be close to the line.

## 2. Check 5: Newton tracking error is about 1.5x PhysX's; the flag is at 2x

| Seed | PhysX success | Newton success | Newton fall-free | Error ratio (Newton / PhysX) |
|---|---|---|---|---|
| 1 | 0.969 | 0.484 | 1.000 | 1.585 |
| 2 | 1.000 | 0.625 | 0.984 | 1.466 |
| 3 | 0.953 | 0.391 | 1.000 | 1.488 |

The largest ratio is 79% of the flag value (2.0). Clean P0 policies already lose about a third to half of their
episode successes on Newton through tracking error alone, without falling. A clean blinded policy with a ratio
somewhat above 1.6 would still be within normal spread; one near 2 would be at the line.

## 3. Check 2: the thresholds moved with the training recipe

The randomized P0 v4 policies are smoother than P0 v2's: they clip motor torque in about 0.2% of steps (v2:
2.6–5.2%) and put less of their action power at high frequency (0.010–0.011; v2: 0.014–0.017). So two thresholds
are much tighter than v2's: actuator limits 0.0034 (v2: 0.078) and action jitter 0.0162 (v2: 0.0251). The actuator
measure is unchanged by the delay motor (computed versus applied effort of the same DC motor model); the
difference is in how the policies act.

Consequence: a policy trained without friction and delay randomization (such as P0 v2) would exceed the v4
actuator-limit threshold on every seed, and one of the three v2 seeds would exceed the jitter threshold. In the
blinded set, a check 2 flag on actuator limits or jitter for a policy trained with less randomization than P0 v4
(for example a policy whose delay is 0) may come from the training recipe rather than from an exploit. Read such
flags together with the policy's training settings when the blinding key is opened.
