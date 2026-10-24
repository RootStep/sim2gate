# v2 phase 2, part A, amendment 1: the final, bounded pilot round (frozen)

Approved by VJ on Oct 24, 2026, after reviewing the 16 pilot robots' measurements (`evidence/phase2/pilots/`). It was
chosen with knowledge of those results, and it is disclosed as such. It changes only which arms are piloted next and how
one symptom is tested; the frozen part A rules otherwise apply unchanged.

## What is piloted

| Arm | Change from P0 v5b | Intended symptom | Seeds |
|---|---|---|---|
| Φ-slip-mild | Training static friction **0.3–0.6** (dynamic = 0.75 × static); slip penalty kept | Slip: foot slip p95 > 0.1185 m/s (part A's threshold) | 219, 220 |
| Φ-delay | No command delay in training (the same change as Φ-jitter) | **Delay fragility:** check 1, scored under the v2 success threshold, flags `action_latency_ms`. The latency cliff is inside the realistic 0–30 ms range | 221, 222 |

There is one candidate per arm and no further candidates. Seeds 219–222 are new; no earlier pilot robot is reused.

**Why these two.**

- **Φ-slip-mild:** low-friction training at 0.2–0.5 did produce slip but failed the success test; 0.3–0.6 is the
  milder setting in between. It is the physics-caused partner for R-slip, the only arm that passed. That pair (same
  symptom, two causes) is what C2 needs.
- **Φ-delay:** v1's no-delay robots (P2) were flagged 3 of 3 by check 1's latency sweep. That was known before these
  pilots. The phase 2 pilots measured only jitter, which was the wrong signature for this change.
- **Reference:** under the v2 threshold, none of the 20 clean robots has a latency cliff inside 0–30 ms.

## Measured, and the pass rule

Each robot is measured for:

- nominal PhysX success over 256 episodes;
- check 1 on the v2 envelope;
- check 2.

An arm passes only if **both** robots qualify (at least 85% over 256 episodes, v2 threshold) **and** both show the
intended symptom, with no config or run problems.

## Hard stop (VJ, Oct 24)

- **If Φ-slip-mild fails,** the planned cause study is closed. C2 and C3 are reported as **not tested**, and v2 moves
  to phase 3 (fix verification) and phase 4.
- **If Φ-slip-mild passes,** the study's arms are Control, R-slip, Φ-slip-mild and, if it passes, Φ-delay. Before part
  B is frozen, its symptom definitions are revisited as patterns across several measures. The pilots showed most
  interventions change several metrics at once.
- **No further pilot rounds** follow this one.

## Dropped, not revisited

R-jitter and R-hop are not piloted again:

- weakening the action-rate penalty barely moved jitter;
- the air-time reward broke walking without producing hopping.

Φ-slip's original candidates stay dropped as recorded.

## Cost

4 robots × about 55 minutes, about 3½–4 GPU hours.
