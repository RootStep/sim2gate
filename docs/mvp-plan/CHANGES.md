# Changes to the frozen MVP plan

The frozen plan (`MVP_PLAN_FROZEN_2026-10-12.md`, SHA-256 `9b9d2c85…dc805677`) is never edited. Each change is
logged here with its date, reason, what it affects, and who approved it. Acceptance criteria A1–A6 are unchanged
unless an entry says otherwise.

## C1 — Who approves plan changes (Oct 13, 2026)

- **Was:** the plan names Vihaan as the approver ("It needs Vihaan's approval before Phase 1 starts; after approval
  it is frozen and any change is logged").
- **Now:** VJ approves changes to the frozen plan from here on. Vihaan's role as lead researcher, and his role as
  holder of the blinding key, are unchanged.
- **Approved by:** VJ, Oct 13, 2026.

## C2 — Checks run on Isaac Lab directly; the Arena port follows validation (Oct 13, 2026)

- **Was:** checks 1, 2 and 5 are built as Isaac Lab-Arena plug-ins on a Go2 embodiment and velocity-tracking task
  ported into Arena. Check 1 uses Arena variation plug-ins and its experiment runner, and check 5 uses Arena's engine
  switch.
- **Now:** for the MVP, all three checks run on Isaac Lab 3.0 directly, on Isaac Lab's own Go2 flat
  velocity-tracking task and its RSL-RL train and play entry points.
  - Check 1 applies its physics variations through Isaac Lab's event terms and configuration overrides.
  - Check 5 uses Isaac Lab's physics selector (`physics=isaacsim_physx` vs `physics=newton_mjwarp`).
  - The Arena port (Go2 embodiment, task, metric and variation plug-ins), and contributing the physics variations
    and exploit metrics back to Arena, come after the MVP validation.
- **Why:**
  - Arena at the pinned commit (2cc9e97) has no Go2 embodiment and no locomotion task, only manipulation tasks.
    Porting them means rebuilding the task with observations that match the trained policies exactly, which is
    several days of work. It also adds a new way to fail parity criterion A5 that has nothing to do with the
    checks themselves.
  - Isaac Lab 3.0 now provides what the plan wanted from Arena for these checks: a PhysX/Newton (MuJoCo-Warp)
    switch on the same task, and event-based physics randomization.
  - The policies, observations, actions and task code stay identical between check runs. Only the measured
    condition changes, which is what check 5's parity conditions require.
- **Affects:**
  - The "Built on Isaac Lab-Arena" section, the sentences in checks 1, 2 and 5 that name Arena plug-ins, and the
    "Decisions for Vihaan" item on building the checks as Arena plug-ins.
  - Check 1's joint sweep (report only) uses its own uniform sampler and analysis instead of Arena's sensitivity
    analysis, with the same uniform-draw requirement.
  - Thresholds, flag rules, the test set, blinding and acceptance criteria are unchanged.
- **Approved by:** VJ, Oct 13, 2026.

## C3 — The actuator-limit threshold is calibrated like the other check 2 metrics (Oct 14, 2026)

- **Was:** "Actuator limits are absolute: any exceedance beyond 1% of steps is flagged."
- **Now:** the actuator-limit metric (share of env-steps where any joint's torque demand is clipped by the actuator
  model or any joint exceeds its speed limit) gets its threshold the same way as the other check 2 metrics. That
  is the largest value across the calibration baselines, P0 seeds 1 to 3, times the safety factor of 1.5. Until a
  threshold is set, the actuator flag reads uncalibrated, neither ok nor flagged.
- **Why:** the three P0 calibration baselines, trained on Isaac Sim PhysX with the stock task (Oct 13), measured
  2.9%, 3.8% and 2.4% under the plan's definition. That was 2.7%, 3.6% and 2.3% of steps with a torque demand
  capped by the Go2 motor model, and 1.0%, 1.2% and 0.8% with a joint past its speed limit. These policies walk
  well, with no falls in 64 episodes each and a mean velocity-tracking error of 0.09 to 0.10 m/s. Brief motor
  saturation in fast steps is normal for the Go2. Under the 1% rule every clean policy would be a no-go, so
  acceptance criterion A4 (at most 1 of 5 held-out clean baselines flagged) would fail because of the threshold,
  not the check.
- **Affects:**
  - The check 2 threshold paragraph and the "Decisions" item on check 2 thresholds, which named the absolute 1%
    rule.
  - Targets P3 and P4 (criterion A1) are unaffected: they are judged on the other metrics.
  - The metric's definition is unchanged.
- **Approved by:** VJ, Oct 14, 2026.

## C4 — P0 is the stock Go2 flat task plus a foot-slip penalty (Oct 14, 2026)

- **Was:** P0, the clean baseline, has no flaw and no config change from default. The test set defines P3 as "Low
  ground friction, foot-slip penalty removed", which presumes a default that has a foot-slip penalty.
- **Now:** P0 is the registered, versioned task variant `Sim2Gate-Velocity-Flat-Go2-P0-v1`
  (`sim2gate/mvp/tasks.py`): Isaac Lab's `Isaac-Velocity-Flat-UnitreeGo2` plus Isaac Lab's own `feet_slide`
  reward term on the four feet.
  - **Weight −0.25.** This is a proposed starting value taken from Isaac Lab's H1 configuration, not a setting
    validated for the Go2.
  - **Versioning.** Any later change is a new version with a new task id.
  - **Training.** Isaac Sim PhysX (`physics=isaacsim_physx`), with the task's own 300 iterations.
- **Baseline acceptance criteria, fixed before retraining.** Each is measured by check 2 on the fixed command set
  (64 envs, one 20 s episode each). A P0 seed that misses any criterion is not used for calibration.
  - **Command tracking:** mean |commanded − actual| base xy velocity ≤ 0.15 m/s. That is about 1.5× the stock
    policies' 0.09–0.10 m/s, and the plan's own tolerance for episode success.
  - **Falls:** at most 1 of 64 episodes ends before its time limit.
  - **Sustained-contact slip:** the 95th percentile of horizontal foot speed is ≤ 0.5 m/s, counted only at
    physics updates where that foot has been in contact for at least 4 updates (20 ms) in a row. 0.5 m/s is about
    half the median commanded speed; the 22 mm foot sphere pivoting on the calf moves its centre by at most about
    0.2–0.4 m/s.
  - These criteria decide acceptance only. The calibrated slip metric stays the plan's
    (`foot_slip_p95_mps`, all in-contact instants).
- **Scope.**
  - Only P0 seeds 1–3 are retrained now, and all five check 2 thresholds are recalibrated from them.
  - P0 seeds 4–8, and P1–P5, are not trained or touched until the checks and thresholds are frozen (plan step 3).
  - P3 will then be this P0 variant with **two** explicit changes: `feet_slide` removed, and low ground friction.
    Its result tests that combined perturbation, not either change alone.
- **Why.** The stock P0 seeds 1–3 (Oct 13) tracked their commands (mean error 0.09–0.12 m/s, no falls), but check 2
  measured foot slip p95 of 3.0–3.2 m/s. A raw trace of seed 1 confirmed the measurement:
  - The foot-to-sensor mapping was correct.
  - The current normal force was used.
  - Feet were within 0.5 mm of the ground when in contact.
  - Reported foot speed matched finite differences of foot position (correlation 0.998).
  - 35% of in-contact instants had a foot moving faster than 1 m/s.
  - Several feet lifted off 0–3 times in 6 s while others stepped 16–43 times.

  So check 2 detected real foot dragging. Whether that is a simulator exploit has not been shown: that would need
  evidence that the policy benefits from simulator behaviour that would not hold physically. The missing slip
  penalty is a plausible explanation, not yet a demonstrated cause. With dragging baselines, the slip threshold
  would have been 4.8 m/s, and P3 could not stand out from P0.
- **Superseded runs preserved.**
  - The stock P0 seeds 1–3 (checkpoints `37b043f4…`, `4ef95e11…`, `49667013…`), their check 2 reports
    (measurement versions 2 and 3), both threshold files, the trace and all logs are kept unmodified:
    - on the instance, in `mvp_runs/P0_stock_superseded`;
    - as `s2g-p0-stock-superseded.tgz` (SHA-256 `3f6e83d1ba4f74d44c42c6bd42915cecfd8261e3cfe9594e411bccc06fa4e310`,
      60 files, verified).
  - They are not used for calibration.
- **Approved by:** VJ, Oct 14, 2026.
- **Recorded by Vihaan:** Oct 14, 2026, before P0 v2 was trained (confirmed by VJ).

## C5 — P0 v2: the same penalty, trained for 1,000 iterations (Oct 14, 2026)

- **Was (C4):** P0 v1, `Sim2Gate-Velocity-Flat-Go2-P0-v1`, trained for the task's own 300 iterations.
- **Now:** P0 v2, `Sim2Gate-Velocity-Flat-Go2-P0-v2`. This is P0 v1 trained for 1,000 iterations. Nothing else
  changes: same `feet_slide` weight (−0.25), same physics, the same acceptance criteria (C4, unchanged) and the
  same seeds 1–3.
- **Why:** first P0 v1 run (Oct 14), measured by check 2 against the C4 criteria.

  | Seed | Checkpoint | Tracking error | Falls (of 64) | Sustained-contact slip p95 | Result |
  |---|---|---|---|---|---|
  | 1 | `8c5926de…` | 0.086 m/s | 1 | 0.128 m/s | accepted |
  | 2 | `28679f1e…` | 0.098 m/s | 15 | 0.324 m/s | not accepted (falls) |
  | 3 | — | — | — | — | not measured: Isaac Sim crashed at shutdown (below) |

  Seed 3 trained all 300 iterations and saved `model_299.pt`. The process then died inside Isaac Sim's
  `SimulationApp.close()` and exited 1, so the driver recorded a training failure and skipped check 2. This crash is
  in Isaac Sim's shutdown, not in Sim2Gate or the policy. For v2 the driver accepts a run that exits non-zero only
  when all of these hold: the log's last training line is the final iteration, the newest checkpoint is the final
  one, and the crash dump shows `SimulationApp.close()` in progress. It records this in the manifest
  (`train_shutdown_crash`). Any other non-zero exit is still a training failure. v1 seed 3 is not measured after the
  fact, since v1 is superseded either way by seed 2's falls.

  The penalty did what it was added for: sustained-contact slip was well within its criterion for both trained
  seeds. The failure was falls. A policy trained with an extra penalty on the stock task's short 300-iteration
  budget (Isaac Lab's rough-terrain Go2 trains for 1,500) being undertrained is the leading explanation, not a
  demonstrated cause. Changing only the training length tests it. A rejected seed is not retrained or replaced
  under another seed number: choosing seeds until they pass would bias the calibration and the held-out
  comparison (A4).
- **If P0 v2 also misses the criteria:** stop and review the evidence before any further change. Do not keep
  adjusting.
- **Superseded runs preserved:** the P0 v1 runs (checkpoints, check 2 reports, logs, manifest) are kept
  unmodified, in `mvp_runs/P0_v1_superseded` on the instance and as `s2g-p0-v1-superseded.tgz` (with the v1 training
  run folders; SHA-256 `673d6d4c7302ab65045281a4221476314719a67e251d3502484046873090ec9e`), and are not used for
  calibration.
- **Approved by:** VJ, Oct 14, 2026.
- **Recorded by Vihaan:** Oct 14, 2026, before P0 v2 was trained (confirmed by VJ).
- **Result (Oct 14):** all three P0 v2 seeds met the C4 criteria (tracking error 0.057–0.064 m/s, no falls,
  sustained-contact slip p95 0.090–0.093 m/s), and the check 2 thresholds were set from them. Details and files:
  `calibration/p0v2/`.

## C6 — P0 v3 (friction and delay randomization); check 5 on the same robot, flagged on falls (Oct 14, 2026)

- **Was:**
  - P0 v2 trains on Isaac Lab's Go2 flat task, which never randomizes ground friction (fixed 0.8) and has no
    action delay. The plan defines P1 as "friction and added-mass randomization set to a near-zero range" and P2 as
    "action delay fixed at 0 in training", which presume a clean baseline that randomizes both.
  - Check 5: flagged if Newton's episode-success rate is below 80% or the Newton/PhysX tracking-error ratio
    exceeds 2.
- **Now:**
  1. **P0 v3**, `Sim2Gate-Velocity-Flat-Go2-P0-v3` = P0 v2 plus:
     - ground friction randomized over the plausible real range, static 0.4–1.2 (dynamic 0.3–0.9, never above
       static);
     - a command delay of 0–30 ms (0–6 physics steps, drawn per episode) in the DC motor model
       (`sim2gate.mvp.actuators.DelayedDCMotor`, the same delay mechanism check 1 sweeps).
     Everything else is unchanged: the foot-slip penalty, the stock ±25% base-mass randomization, 1,000 iterations,
     Isaac Sim PhysX, the C4 acceptance criteria and seeds 1–3. P0 seeds 1–3 are retrained, check 2's thresholds and
     the episode-success reference are recalibrated from them, and checks 1 and 5 are rerun. P1 is then P0 v3 with
     friction and mass randomization near zero, and P2 is P0 v3 with the delay fixed at 0.
  2. **Check 5 runs the same robot on both engines.** PhysX lists the Go2's joints by type (all hips, then thighs,
     then calves) and MuJoCo-Warp leg by leg. Both check 5 runs now use the training engine's joint and body order
     (Isaac Lab 3.0's `joint_ordering` / `body_ordering`), and parity requires identical joint and body names.
  3. **Check 5's first rule counts falls:** flagged if fewer than 80% of Newton's episodes end without a fall, or
     (unchanged) if the tracking-error ratio exceeds 2. Newton's episode-success rate is still reported. The PhysX
     nominal-sanity condition keeps the episode-success definition.
- **Why** (P0 v2 seeds 1–3, Oct 14; files in the `s2g-c15` bundle):
  - Check 1 flagged all three clean baselines on latency, with cliffs at 20–25 ms inside the plausible 0–30 ms. A
    policy trained without delay cannot be told from P2, and A4 would fail on the threshold's premise, not the
    check. Friction cliffs were at 0.33, just outside the plausible range.
  - Check 5 first measured 0% Newton success. Reading the robot back from each engine (`sim2gate.checks.robot_params`)
    showed the joint-order mismatch; Isaac Lab's own transfer guide warns that it makes a walking policy fall within
    a few dozen steps. A policy trained on Newton failed on PhysX in the same way, and neither the motor model nor
    the joint armature explained it.
  - With matching joint order, no P0 v2 episode fell on Newton (0 of 192), and the error ratio was 1.34–1.48. Yet
    only 55–72% of episodes met the episode-success tracking limit (1.5 x the PhysX baseline's error), because that
    limit is stricter than the ratio rule's 2x: the ratio rule could never decide. Falls are what the plan's P5
    ("contact settings Newton will not reproduce") is meant to produce; tracking stays judged by the ratio.
- **Affects:** the test-set rows for P0, P1 and P2; check 5's parity conditions and flag rule. Thresholds are
  recalibrated from P0 v3; acceptance criteria A1–A6 are unchanged.
- **Superseded runs preserved:** the P0 v2 baselines, their check 2 calibration (`calibration/p0v2/`), and their
  checks 1 and 5 runs and diagnostics are kept unmodified on the instance and as archives, and are not used for
  calibration.
- **Approved by:** VJ, Oct 14, 2026.
- **Recorded by Vihaan:** pending.
