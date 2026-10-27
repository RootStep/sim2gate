# Diagnostic Benchmark v0.1 — Development Pilot Protocol

**Status: DRAFT FOR APPROVAL — October 27, 2026. Not a training authorization, protocol freeze, or confirmatory study.**

## 1. Objective and boundary

Determine whether fresh Unitree Go2 Isaac Lab/RSL-RL training runs can (a) produce viable policies, (b) apply H0/T1/D1/D2 interventions exactly as intended, (c) exhibit independently measured behavioral manifestations, and (d) supply credible runtime, qualification, and variance estimates for planning the blinded benchmark. This pilot cannot establish Sim2Gate superiority, hardware transfer, or confirmatory diagnostic accuracy.

Source protocols: PROTOCOL.md, PROTOCOL_DECISIONS.md, FLAW_RECIPES_DRAFT.md. All unresolved decisions there remain unresolved unless separately approved.

## 2. Entry criteria (hard stop before GPU use)

- Gates 1–5: user-reported PASS (structural 7/7, real config 4/4, semantics 10/10, fake-buffer actuator 18/18, real Isaac Lab DelayBuffer with minimal motor 165/165). Preserve raw reports, exact code commit, runtime and environment fingerprint. These gates do **not** establish complete live robot/actuator integration.
- Verify the full Go2 environment instantiates the intended **real** DelayedDCMotor actuator and executes at least one bounded nontraining reset/step, with logged sampled delays and command effect. Check the ramp configuration and the evaluation play-mode override. Abort on mismatch.
- Resolve pilot-only operational definitions of nominal qualification, independently measured manifestation, commands, disturbance grid, repeat count, missing records, and allowed retries; do not reuse v2 thresholds uncritically.
- Obtain explicit approval of run IDs/seeds, GPU-hour ceiling, per-run timeout, storage destination, and stop conditions. Do not touch the active Brev checkout or untracked secrets; use the isolated benchmark checkout.
- Ensure prediction/truth schema and leak-proof IDs exist before any diagnostic comparisons. No benchmark freeze is implied.

## 3. Candidate arms and interpretation

| Arm | Intervention | Configuration evidence | Independent behavioral question | Caveat |
|---|---|---|---|---|
| H0 | Unmodified P0 v5b baseline | No env or agent diff | Meets predeclared nominal qualification? | No intervention does not guarantee healthy behavior |
| T1 | Reduce feet_slide weight from -0.25 to -0.025 | Reward weight diff, term contribution logs | More sustained slip/slip p95 under standardized commands/terrain? | Reduction does not guarantee slip manifestation |
| D1 | Static friction range (0.35,1.2) to (0.3,0.6), dynamic ratio 0.75 | DR config diff, actual friction sample histogram | Worse performance at specified out-of-training-range friction? | **Lower-and-narrower**, not an isolated width intervention; width-only comparator (0.625,0.925) is exploratory, not a fifth approved arm |
| D2 | Set max actuator delay from 6 to 0 physics steps | Config, live sampled lag, training ramp logs | More vulnerable to held-out delay stress? | H0 delay ramp starts at round 500 and reaches full range at round 800; early training has no H0/D2 delay contrast |

## 4. Size and assignment

**Proposed minimum screening allocation:** four independent fresh training runs: H0×1, T1×1, D1×1, D2×1. Preassign unique random seeds before training, with no reuse of historical development/confirmatory seeds. One run per arm is a smoke test only: it cannot estimate across-seed effects or provide robust causal comparisons. If budget allows after reviewing the first four, prioritize a second H0 and paired additional seeds, under a separately approved amendment. Do not silently expand or substitute runs.

Use common training code, iteration cap, evaluation schedules, hardware and base hyperparameters, except intentional recipe changes and harmless experiment identifiers. Record all deviations. Treat training run as the independent unit; do not count episodes as independent policies.

## 5. Run sequence

**Stage A — Nontraining integration gate:** fingerprint Isaac Lab/Isaac Sim/RSL-RL, torch, GPU driver, repository commit, environment and actuator config; inspect one bounded Go2 reset/step with actual motor class and sampled delays. Capture evidence before GPU training authorization.

**Stage B — Training feasibility (only after approval):** execute the four preassigned runs serially or under a documented resource schedule; preserve logs and checkpoints including unsuccessful attempts. Record wall-clock, GPU-hours, peak VRAM, training stability, checkpoint integrity and training completion.

**Stage C — Independent evaluation:** run a predeclared nominal command set and disturbance grid (friction, delay and standardized slip-relevant conditions). Use identical commands, episode seeds, rollout counts and compute budgets across arms where meaningful. The evaluation operator must not derive truth from Sim2Gate alerts. Record fall, tracking error, sustained slip/slip p95 and sensor/telemetry completeness with units and definitions.

**Stage D — Development diagnostics (optional, not a claim):** evaluate frozen-in-pilot Sim2Gate and matched-budget baselines on pilot policies only. Record inputs, outputs, coverage, runtime and obvious leakage; use results solely to improve protocol before freeze. Do not present these as held-out benchmark results.

## 6. Outcomes and denominators

Report **separately**: intervention applied (config + runtime evidence); policy trained and checkpoint valid; nominal qualification (predeclared); manifestation present/absent/indeterminate (independent evaluation); and diagnostic eligibility. Never equate planted intervention with manifested defect or unique causal mechanism. A nonmanifesting planted policy stays in the assigned-arm denominator. Preserve crashes, timeouts, missing metrics, retries, and excluded runs with reason codes. No post-hoc replacement of a failed seed.

Pilot readiness metrics: per-arm execution completion, configuration integrity, qualification, measurable stress response, telemetry coverage, runtime and storage cost. Report raw observations and uncertainty honestly; with one policy per arm do not infer treatment effects.

## 7. Proposed stop conditions

Stop and escalate if: live actuator integration disagrees with config; unintentional environment/agent differences appear; seed/checkpoint/label leakage is detected; required ground-truth telemetry is missing; any run exceeds its approved time/GPU budget; a recipe is a no-op; the evaluation truth is computed using Sim2Gate's own decision threshold; or a safety/operational problem arises. Log all partial artifacts and decisions. Exact numeric budgets and qualification/manifestation thresholds are **OPEN FOR APPROVAL**, not invented here.

## 8. Required artifacts

- PILOT_APPROVAL.md with named approver, approved run list/seeds, GPU cap, runtime limits and start authorization
- Immutable per-run config, Git commit, environment fingerprint, config SHA-256 and checkpoint SHA-256
- Training logs, reward terms, actuator-delay samples and friction samples
- Independent evaluation specification and per-episode results, with missingness and units
- Intervention/manifestation/qualification adjudication ledger, including failures
- Runtime/GPU-hours/storage accounting and protocol deviations
- PILOT_REVIEW.md with feasibility findings, recipe decisions, revised sample-size assumptions and next-gate recommendation

## 9. Explicit exclusions

No confirmatory cohort, no sealed test-set access, no claim of superiority over Arena-style sensitivity, raw signals or success-only baselines, no causal attribution solely from planted recipe, no hardware sim-to-real claim, no automatic expansion to 40+40, and no protocol freeze until independent review and explicit approval.

## 10. Decisions needed before execution

1. Approve four-run allocation or replace with the original two-H0/two-flawed design (which cannot screen all three flawed recipes).
2. Select exact seeds, training iteration cap, compute/storage ceilings and retry policy.
3. Freeze **pilot-only** nominal qualification and independent manifestation grid/thresholds (distinct from confirmatory freeze).
4. Confirm nontraining live Go2 actuator integration evidence and separate custody of intervention labels.
5. Explicitly authorize GPU pilot after the above evidence is reviewed.
