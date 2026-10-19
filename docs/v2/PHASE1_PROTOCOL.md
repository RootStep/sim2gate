# v2 phase 1: protocol (frozen before any phase 1 training)

Phase 1 trains the v2 clean robot, P0 v5. These robots set Layer 1's limits and serve as Layer 2's control arm (v2
plan, "Phases"). Driver: `sim2gate.v2.phase1`. Task: `sim2gate.v2.tasks`. The SHA-256 of this file is recorded in
`PHASE1_FREEZE.md` before the smoke test runs.

## 1. P0 v5 (VJ's decision, Oct 19, 2026: train across the full envelope)

P0 v4 (foot-slip penalty; 0–30 ms command delay ramped in over rounds 500–800; 2000 rounds), plus:

| Condition | v1 / P0 v4 training | P0 v5 training | v2 envelope (check 1) |
|---|---|---|---|
| Added payload on the base | none (stock base-mass scale 0.8–1.25) | **0–8 kg uniform per robot** (startup), on top of the stock scale | 0–8 kg, swept to 12 |
| Ground friction (static) | 0.4–1.2 | **0.35–1.2** (dynamic 0.2625–0.9) | 0.35–1.2, swept to 0.125 |
| Motor strength | 1.0 | **0.8–1.0 per episode**, on the delay's ramp | 0.8–1.0, swept to 0.7 |
| Command delay | 0–30 ms (ramp) | unchanged | 0–30 ms, swept to 45 |
| Sensor noise | task default | unchanged | 1–2×, swept to 2.5× |

**Why the full envelope.** The plan's text said "P0 v4 + payload 0–8 kg". VJ chose the full envelope because v1's
eight clean robots already lost walking between motor strength 0.775 and 0.825, which is inside v2's 0.8–1.0 range.
Training on payload alone would likely get clean robots flagged, which is the lesson of v1's change C6.

**Evaluation.** Check 1's nominal profile removes the stock mass event and the payload event, and it replaces the
motor. Every robot is therefore measured at nominal, then swept one condition at a time. The v1 grid and its hash
are unchanged; the v2 envelope is a separate grid (`robustness.PARAMS_V2`, recorded in every sweep). Check 2 runs
the task's own play mode, so robots carry their training payload and motor strength there, exactly as P0 v4 robots
carried their mass scale and delay in v1.

## 2. Smoke test

The smoke test is seed 99, 10 rounds, with the full training record and all four measurements. It must show:

- every measurement wrote its output;
- the sweep is on the v2 envelope;
- the saved training config records payload 0–8 kg and motor strength 0.8–1.0;
- the nominal profile removed the payload.

The smoke robot is not expected to walk.

## 3. Pilot (seeds 1 and 2)

Each pilot robot is trained, then measured:

- **nominal:** PhysX, 256 episodes (64 commands × 4);
- **check 1:** swept over the v2 envelope;
- **check 2;**
- **Newton nominal:** 256 episodes.

**The pilot passes only if both seeds**:

1. qualify: nominal success **≥ 85% over 256 episodes**; and
2. have **no check 1 flag inside the v2 envelope**: a scored sweep with no parameter's cliff (first value below 80%
   success) inside its plausible range.

**Success threshold** for both rules: v1's frozen threshold, `frozen_success_p0v4.json` (1.5 × P0 v4's nominal
tracking error). The v2 threshold is recalibrated from the 20 clean robots afterwards, and is not used to judge
them.

**If the pilot fails,** P0 v5 is not frozen. The recipe is revised as a new version (P0 v5b, logged with its
reason, approved by VJ) and piloted again. Candidate revisions, in order:

- ramp the payload in like the delay;
- more training rounds;
- a narrower range for the parameter that failed.

A narrower range would be reported as a change to the envelope.

**If it passes,** P0 v5 is frozen as is.

## 4. The 20 clean robots

Seeds 101–120, each trained with the full training record (`EvidenceRecorder`, as in phase 0A) and the monitor, and
measured like the pilot robots.

A robot that does not qualify (< 85% over 256) is not measured further and is replaced by the next spare seed
(121–125). It stays in the record with its result. More than 5 failures stops the run for a decision.

Check 1 flags on clean robots are **recorded, not used to reject them**. Their rate is what Layer 1's false-alarm
estimate is about.

## 5. What phase 1 delivers, and what it does not

**It delivers:** 20 qualified clean robots with:

- their training records;
- nominal (PhysX and Newton), check 1 v2 and check 2 measurements;
- a bundle with a SHA-256 manifest.

**It does not deliver:** Layer 1's v2 limits. These include the combined check 2 limit, the v2 success threshold
and check 5's parity reference. They are computed from these robots by rules written and frozen in a separate step
before any test robot is trained (v2 plan, "Layer 1: setting the limits").

## 6. GPU budget

The plan allows 23 GPU hours. Estimate:

- smoke test: 0.3 h;
- pilot: 2 robots × about 1 h each;
- clean robots: 20 × about 55 min each, about 18–19 h.
