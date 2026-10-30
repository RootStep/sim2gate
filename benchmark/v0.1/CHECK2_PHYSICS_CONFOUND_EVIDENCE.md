# Check 2 physics confound in the historical v2 Phase 2 measurements — evidence note

**Status: DEVELOPMENT EVIDENCE NOTE — 2026-10-30.** Records a methodological finding. It changes no frozen v2 document or result. Authorizes no evaluation, rerun, training or freeze.

## Finding

In the v2 Phase 2 study (code `e502dae`, Oct 25–26, 2026), **Check 2 measured each robot in its own training task.** The arms were therefore measured under different evaluation physics:

| Robots | Check 2 task (recorded `play_args`) | Ground friction, static (dynamic) | Command delay |
|---|---|---|---|
| Controls p0v5b_s101–s105 | `Sim2Gate-Velocity-Flat-Go2-P0-v5b` | 0.35–1.2 (0.2625–0.9) | 0–6 physics steps (0–30 ms) |
| R-slip s301–s305 | `Sim2Gate-Velocity-Flat-Go2-P2-R-slip-k0.1` | 0.35–1.2 (0.2625–0.9) | 0–30 ms |
| Φ-slip-mild s306–s310 | `Sim2Gate-Velocity-Flat-Go2-P2-Phi-slip-mild-mild` | **0.3–0.6 (0.225–0.45)** | 0–30 ms |
| Φ-delay s311, s312, s314, s315 | `Sim2Gate-Velocity-Flat-Go2-P2-Phi-delay-nodelay` | 0.35–1.2 (0.2625–0.9) | **none** |

s313 has no measurement.

## Evidence

**Recorded task per report (CONFIRMED).** Read on 2026-10-30 by the operator from `play_args` in all 19 `check2.json` reports. They live in the unsealed study tree on the shared PC: `controls/p0v5b_s10N/check2.json` and `robots/study_s3NN/measure/check2.json`. Every report has `terrain_type: plane` and `physics_context.overrides = ['physics=isaacsim_physx']`. No report carries a friction or delay override.

**Physics per task (CONFIRMED from code at `e502dae`).** None of these files changed between `e502dae` and this note's base commit.
- P0 v5b friction: `src/sim2gate/v2/tasks.py:22-23`.
- P0 v5b command delay of 0–6 physics steps, inherited from P0 v3/v4: `src/sim2gate/mvp/tasks.py:46`.
- Φ-slip-mild friction (0.3–0.6 static, dynamic 0.75×) and Φ-delay `max_delay = 0`: `src/sim2gate/v2/phase2_arms.py:31-34, 65-77`.

**Execution path (CONFIRMED from code at `e502dae`).**
1. The study driver measures each robot with its training task: `src/sim2gate/v2/phase2_study.py:63` → `phase1.train_and_measure(..., task=...)` (`src/sim2gate/v2/phase1.py:149-176`).
2. `measure_commands` (`phase1.py:64-89`) runs Check 2 as `sim2gate.checks.isaaclab_exploits --task <training task>` with only the physics-engine override (`phase1.py:81-83`). `isaaclab_exploits.py` applies no evaluation profile.
3. By contrast, nominal, sweep (Check 1) and Newton (Check 5) run through `sim2gate.checks.isaaclab_conditions`. Its `apply_profile` (`isaaclab_conditions.py:102-129`) does three things before the environment is built (hook at `:215-217`, refuse-if-not-applied at `:250-251`):
   - fixes friction at 0.8 static / 0.6 dynamic;
   - removes the listed randomization events;
   - replaces the actuators with the swept DC motor, which has no delay unless the delay is being swept.

**Runtime check for nominal, Check 1 and Check 5 (CONFIRMED for all 19 policies).** All 57 reports carry the identical `profile: sim2gate-nominal-v1` and `profile_record`, as read on 2026-10-30 from the transferred measurement archives. That is 14 study robots and 5 controls × `nominal.json`, `sweep.json` and `newton.json`. The shared record is: `contact_material` static 0.8 / dynamic 0.6; actuators `base_legs: SweptDCMotor`; removed events `add_base_mass`, `add_payload` and `base_com`. Their recorded tasks remain the training tasks; the profile overrides the friction, events and motor model.

**Limitation of the evidence.** Check 2 records the task, not the friction or delay values actually drawn. The values above are the configured ranges for each recorded task, read from code at the study commit.

## Affected and unaffected results

- **Historical Check 2 metrics for Φ-slip-mild and Φ-delay are confounded** relative to controls. Φ-slip-mild was measured on lower-friction ground; Φ-delay with no command delay. Differences in slip, jitter, actuator-limit share, penetration, contact force and hop measures cannot be attributed to the policy alone.
- **R-slip vs controls (Check 2): matched evaluation physics.** The policies differ by the intended reward intervention, which is the effect under study, not a confound.
- **v2 Phase 2 approach A (behaviour only), interpretation qualified.** 7 of A's 15 primary features are Check 2 metrics (`docs/v2/evidence/phase2/scoring/features.json`, columns `A:check2:*`). The recorded results stand as historical results: 17/19 cause identification and 8/10 slip-arm separation. Their reading as policy-only behavioural discrimination is weakened for the Φ-slip-mild and Φ-delay arms. The Phase 2 leak audit checked configuration leakage into features, not evaluation-physics parity.
- **Nominal success, Check 1 and Check 5: not affected** for any of the 19 policies (runtime-confirmed above).

## Required consequences

1. Any new or corrected evaluation runs all policies in **one common evaluation task** under stated fixed conditions, never a policy's training task.
2. Historical Check 2 outputs are kept as a **historical comparator**, labelled "measured under each arm's training physics".
3. A common-physics Check 2 rerun is a **post hoc corrected retrospective** comparator, not an original or independent result.
4. Future studies: add an evaluation-physics parity check (task, friction, delay and events identical across arms) to the leak audit.
