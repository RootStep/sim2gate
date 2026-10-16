# Test set v1: the planted-flaw policies and the blinded run (Oct 15, 2026)

Fixed before any planted or held-out policy is trained (plan, step 3). **Approved by VJ, Oct 15, 2026**: the P1–P5
definitions below and the rule that each policy is checked in its own task. **P5's ground revised and approved by VJ,
Oct 16, 2026** (stiffness 2000, fallback 3000), from the stand pilot below. Code: `sim2gate.mvp.tasks` (the
definitions), `sim2gate.mvp.blinded` (the run), `sim2gate.mvp.verdict` and `sim2gate.mvp.acceptance` (the scoring).

## The policies

Every test-set policy is P0 v4 (C7) with at most one change: the same task, rewards, randomization, delay ramp
(none before round 500, full 0–30 ms from round 800), 2,000 training rounds, PPO settings and Isaac Sim PhysX.

| ID | Flaw (plan) | The one change from P0 v4 | Target | Seeds |
|---|---|---|---|---|
| P0 | none | none | all should pass | 1–8 (1–3 calibrate; 4–8 held out) |
| P1 | narrow randomization | friction 0.78–0.82 static (0.585–0.615 dynamic), instead of 0.4–1.2; base-mass scale 1/1.01–1.01, instead of the task's 1/1.25–1.25 | check 1 (friction or mass) | 1–3 |
| P2 | no latency | command delay 0 throughout (the same delay motor, range 0–0), instead of 0–30 ms | check 1 (latency) | 1–3 |
| P3 | sliding exploit | ground friction 0.2–0.4 static (0.15–0.3 dynamic), and no foot-slip penalty | check 2 | 1–3 |
| P4 | jittery | no action-rate penalty | check 2 | 1–3 |
| P5 | engine overfit | a soft ground contact PhysX models and Newton does not reproduce: PhysX compliant contact in acceleration units, stiffness 2000, damping critical (89.4); fallback 3000 (below) | check 5 | 1–3 |

Task ids: `Sim2Gate-Velocity-Flat-Go2-P1-v1` … `-P5-v1`; P0 is `Sim2Gate-Velocity-Flat-Go2-P0-v4`. 23 policies.

- **P5's mechanism.** Isaac Lab documents the acceleration-spring flag as PhysX-only ("not consumed by Newton").
  The Oct 15 definition (stiffness 1000) was checked on the GPU on Oct 16 with a standing robot, no training and no
  check involved (`evidence/pre-freeze/`): its ground differed between the engines as intended, but the robot sank
  8.7 cm on PhysX, against the ~1 cm assumed. VJ chose to stiffen it; a stand pilot over stiffness 2000–10000
  (`evidence/p5-pilot/`) then showed that the 2 cm sink limit set beforehand and the purpose of P5 conflict:

  | Stiffness | PhysX sink | Standing-height gap, PhysX vs Newton |
  |---|---|---|
  | normal ground (P0) | – | 1.8 cm (Newton lower) |
  | 1000 | 8.7 cm | 6.0 cm |
  | 2000 | 5.4 cm | 3.1 cm |
  | 3000 | 4.2 cm | 2.1 cm |
  | 4000 | 3.6 cm | 1.5 cm |
  | 10000 | 2.1 cm | 0.2 cm |

  The sink levels off near 1.3 cm (2 cm would need about 12,000), and from about 4000 the engine gap is smaller than
  normal ground's, so P5 would no longer be an engine-overfit policy. **Decision (VJ, Oct 16): stiffness 2000**,
  damping critical (2√k). **Fallback, fixed now:** if any P5 seed fails the walking check at 2000, all three P5 seeds
  are trained at 3000 (task `Sim2Gate-Velocity-Flat-Go2-P5-v2`). Nothing about P5 is tuned on check 5's output.
- **A weakened recipe is not a weak policy.** Each planted policy is trained once per seed with these settings.
  Whether its flaw makes it detectably weak is what the run finds out; no setting is changed afterwards to make a
  label come true. If a target check misses its policy, the criterion fails and is reported (`evidence/
  planted-bugs/README.md` notes why P1 may be the hardest).
- **Manipulation check (plan).** Each planted policy must still walk under nominal conditions: Vihaan checks this
  on the unblinded checkpoints before blinding (check 2's nominal tracking and falls, as for the baselines). A
  policy that never learned to walk is retrained with a new seed, and the retraining is recorded in the key.

## Where each policy is checked

Each policy is checked in its own task, in play mode: the simulator it was trained in, as a user's policy would
be. This is what makes the planted flaws visible at all: P5's soft contact is part of its task, so check 5 runs
P5's task on both engines; P3's low friction is part of its task, so check 2 measures it there. Checks 1 and 5
apply the same nominal profile on top of every task (friction 0.8, no mass or centre-of-mass randomization,
observation noise on, no pushes). The calibration (P0 v4 in its own task) was measured the same way.

## The blinded run

1. **Training and blinding, unattended** (VJ's proposal, Oct 16; `python -m sim2gate.mvp.testset`). Vihaan makes an
   RSA key pair on his own computer once and gives only the public key. `testset train` then trains the 20 new
   policies (P0 seeds 4–8, P1–P5 seeds 1–3) in a secret random order, with all output written to files; checks each
   one walks (at least 80% of 64 nominal PhysX episodes succeed, as A5; a failure is retrained with seed + 10, then
   + 20, and three failures stop the run; a P5 failure first moves all P5 seeds to the 3000 fallback); blinds all 23
   with `blinded prepare` (the 20 plus the three calibration baselines). Each policy gets a random ID; its checkpoint
   is copied to `blind/<ID>/model.pt`; `blind/manifest.json` holds only IDs and checkpoint hashes;
   `blind/aliases.json` maps each ID's task alias (`Sim2Gate-Blind-<ID>`) to an opaque token of its task. The key, the
   training logs and settings, the final checkpoints and the walking-check results are then encrypted with Vihaan's
   public key (RSA-OAEP over a random AES-256 key) and deleted, along with the labelled training folders. Only
   Vihaan can open them (`testset unseal`). The console never names a policy.
2. **The runner** runs `blinded run`: for each ID, check 2, check 1's sweep, and check 5's PhysX and Newton runs,
   plus the report-only joint sweep, all under the alias, then the verdict (and its HTML report) with the frozen
   thresholds and success reference shipped in the package
   (`frozen_thresholds_p0v4.json`, `frozen_success_p0v4.json`; a test pins them to `calibration/p0v4`). Logs and
   records name only the ID.
3. **Seal.** When every ID has a complete verdict, `blinded seal` writes `verdicts.sha256` and prints one digest.
   The runner sends the digest to Vihaan; after sealing, `run` refuses to change any verdict.
4. **Replays (A6).** `blinded replay` reruns 3 IDs, chosen by a seeded draw (seed 20261015) from scratch.
5. **Key.** Vihaan unseals and releases the key only after he has the digest. `python -m sim2gate.mvp.acceptance` scores the
   sealed verdicts and the replays against it (A1–A6). Anyone can check the verdicts against the digest.

What blinding protects against is a person adjusting anything while knowing which policy is which. The scoring
itself is automatic and frozen; the blind folder names no policy, and its alias tokens do not either, though
someone set on decoding them could (there are only nine task ids). The runner agrees not to try.

## Before the freeze (GPU, one session)

- The five planted tasks build and run on PhysX (a standing robot on each).
- P5's ground behaves differently on PhysX and Newton (above): done Oct 16, then revised (2000, fallback 3000).
- A known-answer dry run of the whole blinded pipeline on 4 policies whose answers are known: P0 v4 seeds 1–3
  (expected go) and P0 v2 seed 1 (trained without delay; expected no-go, latency), with prepare, run, seal and
  replay. This tests the live chain (GPU measurement → scoring → verdict), not just saved files.
