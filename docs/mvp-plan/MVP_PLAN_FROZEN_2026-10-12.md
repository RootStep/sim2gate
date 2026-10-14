# Sim2Gate MVP: Design and Validation Plan

Oct 8, 2026 · @VJ

This plan defines the first three Sim2Gate checks and how we will prove they work, with criteria fixed before any code runs, in the style of the BSRS preregistered plans. It needs Vihaan's approval before Phase 1 starts; after approval it is frozen and any change is logged.

## Built on Isaac Lab-Arena

Sim2Gate's evaluation checks run inside NVIDIA's open-source Isaac Lab-Arena instead of beside it; Sim2Gate's own contribution is training-time diagnosis. This follows an audit of Arena's main branch on Oct 8, 2026 (commit 66baca7).

| Arena already has | Sim2Gate adds |
| --- | --- |
| Variation plug-ins: lighting, camera, background images, object mass, object disappearing | Physics variations for locomotion: ground friction, robot base mass, motor strength, action latency, sensor noise (check 1) |
| Metric plug-ins: success rate, object moved, joint moved | Exploit metrics: foot slip, ground penetration, contact-force spikes, actuator-limit violations, action jitter (check 2) |
| Two physics engines: PhysX and Newton (MuJoCo-Warp solver) | A transfer comparison between the two engines (check 5) |
| Sensitivity analysis over per-episode results | A Go2 velocity-tracking task and embodiment, which Arena lacks |
| Loading trained RSL-RL policies for evaluation | Training-time diagnostics: reward attribution and Vihaan's V1.4 statistic, linked to Arena results (Experiment T) |

The physics variations and exploit metrics are generic, so we plan to contribute them back to Arena; the training-time diagnostics stay Sim2Gate's own.

## Scope and definitions

The MVP covers one stack, one robot, one task and three checks; everything else waits for v0.2.

| Item | Choice |
| --- | --- |
| Simulator and trainer | Isaac Lab with RSL-RL PPO (current release; version pinned in the run log) |
| Evaluation framework | Isaac Lab-Arena (version pinned in the run log) |
| Robot | Unitree Go2, added to Arena as an embodiment |
| Task | Flat-terrain velocity tracking, ported from Isaac Lab's built-in task into Arena |
| Engines (check 5) | PhysX (training engine) and Newton (MuJoCo-Warp solver) |
| Checks | 2 simulator exploits, 1 robustness envelope, 5 sim-to-sim |

**Episode success** (used by checks 1 and 5): the robot completes a 20-second episode without a fall termination, and its mean linear-velocity tracking error is at most 1.5 times the clean baseline's mean error under nominal conditions. The 20 s length and the 1.5x factor are proposals for Vihaan to confirm.

**Velocity commands:** one fixed command set used everywhere, sampled once with a recorded seed (forward, lateral and yaw rates within the task's training range).

## Test set

We train policies with known, planted flaws before writing any check, so each check is judged against an answer key, the same idea as the planted bugs in the CleanRL and SB3 gates.

| ID | Policy | Planted flaw (config change from default) | Target check |
| --- | --- | --- | --- |
| P0 | Clean baseline | None | All should pass |
| P1 | Narrow randomization | Friction and added-mass randomization set to a near-zero range | 1 |
| P2 | No latency | Action delay fixed at 0 in training | 1 |
| P3 | Sliding exploit | Low ground friction, foot-slip penalty removed | 2 |
| P4 | Jittery | Action-rate penalty removed | 2 |
| P5 | Engine overfit | PhysX contact settings that Newton will not reproduce (for example very soft contacts) | 5 |

- **Seeds:** P1 to P5 train with seeds 1 to 3. P0 trains with seeds 1 to 8: seeds 1 to 3 calibrate thresholds, seeds 4 to 8 are held out to measure false alarms.
- **Total:** 23 policies. Every checkpoint is saved with its exact config and a SHA-256 hash. All training uses PhysX; P5 is designed to work on PhysX but not on Newton.
- **Manipulation check:** each planted policy must still train to the success threshold under nominal conditions. A flawed policy that never learned to walk tests nothing and is retrained.
- **Blinding:** Vihaan assigns each checkpoint a random ID and keeps the key. Whoever runs the checks sees only IDs until all verdicts are recorded.

## Check 2: simulator exploits

Check 2 asks whether the policy relies on physics the real Go2 cannot reproduce. Each metric is an Arena metric plug-in (a recorder term plus an offline compute function), run under nominal conditions with the fixed command set (64 parallel episodes).

| Metric | How it is measured | Suggested fix if flagged |
| --- | --- | --- |
| Foot slip | Horizontal foot speed while that foot is in contact (normal force above a small threshold); 95th percentile | Add or raise the foot-slip penalty; widen friction randomization |
| Ground penetration | Deepest foot or body position below the ground surface, beyond a tolerance | Check contact settings; lower stiffness exploits |
| Contact-force spikes | Peak foot normal force divided by robot weight; 99th percentile | Penalize impact forces; reduce max commanded speed |
| Actuator limits | Share of steps where any joint torque or velocity exceeds the limits in the Go2 model | Penalize torque; clip actions to feasible range |
| Action jitter | Share of action-signal power above one quarter of the control frequency | Add or raise the action-rate penalty |

**Thresholds:** for each metric, the threshold is the largest value seen across calibration baselines P0 seeds 1 to 3, times a safety factor of 1.5 (proposed). Actuator limits are absolute: any exceedance beyond 1% of steps is flagged.

**Output:** each flagged metric becomes a risk with its measured value, the threshold, the margin, a plot and the suggested fix.

## Check 1: robustness envelope

Check 1 finds where the policy breaks as physical conditions move past what it saw in training, and flags it when that point falls inside conditions a real Go2 can meet.

**Method:** each parameter below is an Arena variation plug-in applied at reset. Sweep one parameter at a time, all others at nominal, with Arena's experiment runner: 10 evenly spaced values from nominal to the edge of the plausible real range plus 50%, 64 episodes per value. The **cliff** is the first value where the success rate drops below 80%.

| Parameter | Plausible real range (proposed) | Notes |
| --- | --- | --- |
| Ground friction | 0.4 to 1.2 | Covers smooth floors to rubber mats |
| Added base mass | 0 to 3 kg | Payloads and battery variation |
| Motor strength scale | 0.85 to 1.0 | Battery sag and motor wear |
| Action latency | 0 to 30 ms | A deployed Unitree G1 reported 18 to 30 ms command latency ([RoboGhost](https://arxiv.org/pdf/2510.14952)); confirm for Go2 |
| Observation noise scale | 1x to 2x the task's default noise | Real IMU and encoder noise |

**Flag rule:** a parameter is flagged when its cliff lies inside its plausible real range. The risk reports the cliff value, the training range and the fix: widen randomization to cover the plausible range, then retrain.

**Also run (report only, not a flag in the MVP):** a joint sweep of all five parameters through Arena's sensitivity analysis, to show interactions between them. The sweep must sample uniformly, since Arena's analyzer assumes a uniform draw.

## Check 5: sim-to-sim transfer

Check 5 evaluates the same policy under Arena's two physics engines, PhysX (the training engine) and Newton with its MuJoCo-Warp solver, and compares them. Arena's own documentation warns that behavior can differ significantly between engines, which is exactly what this check measures. Using Arena's engine switch replaces the separate MuJoCo harness in the earlier plan: the policy, observations, actions and task code stay identical, and only the physics changes. A standalone MuJoCo comparison can be added later if partners ask for it.

**Parity checks (all must pass, per policy):**

1. **Same configuration:** the run log shows identical task, embodiment, policy checkpoint, command set, control rate and seeds for both engines; only the physics setting differs.
2. **Nominal sanity:** the policy meets the episode-success definition on PhysX before Newton is run; otherwise the check reports "policy not ready" rather than a transfer result.
3. **Engine settings recorded:** Newton solver parameters are Arena's defaults, pinned by version, and never tuned per policy, so no policy can be made to pass by adjusting the engine.

**Transfer metrics (closed loop, fixed command set, 64 episodes per simulator):**

- Success rate on Newton vs PhysX
- Ratio of mean velocity-tracking error, Newton divided by PhysX

**Flag rule (proposed):** flagged if Newton success rate is below 80%, or the tracking-error ratio exceeds 2. The suggested fix is to review contact parameters and randomize them, then retrain.

## Report and verdict

The verdict is no-go if any check flags a risk; otherwise go. The MVP has no partial scores, because a single readiness score needs real-hardware data to calibrate.

- **Each risk lists:** check, metric, measured value, threshold, signed margin, evidence (plot or sweep table), likely cause and suggested fix.
- **Each run records:** policy hash, config hash, Sim2Gate version, simulator versions, seeds and the command set, so any result can be replayed exactly.
- **Formats:** JSON for machines, HTML for people.
- **Exit codes:** 0 go, 1 no-go, 2 missing inputs, 3 check invalid (a check 5 parity condition failed or the policy did not meet nominal success, so that check is not scored and the report says why).

## Validation protocol

The MVP passes only if every criterion below holds. Criteria are fixed at approval and not changed afterwards; any failure is reported with a diagnosis, as in the CleanRL and SB3 gate reports.

**Procedure:**

1. Freeze this plan: record the approval date and the plan's SHA-256 hash.
2. Build the checks and set thresholds using only P0 seeds 1 to 3.
3. Freeze the code: tag the version and hash it before touching any planted or held-out policy.
4. Run all checks on all 23 policies, blinded.
5. Record verdicts, then unblind and score against the answer key.

**Acceptance criteria (proposed):**

| # | Criterion | Pass if |
| --- | --- | --- |
| A1 | Check 2 detects its targets | Flags P3 and P4 in at least 2 of 3 seeds each |
| A2 | Check 1 detects its targets | Flags the friction or mass parameter for P1, and latency for P2, in at least 2 of 3 seeds each |
| A3 | Check 5 detects its target | Flags P5 in at least 2 of 3 seeds |
| A4 | Few false alarms | At most 1 of the 5 held-out baselines (P0 seeds 4 to 8) gets a no-go |
| A5 | Harness is valid | Check 5 parity conditions hold for every policy, and every planted policy meets nominal success on PhysX |
| A6 | Results replay | Rerunning 3 randomly chosen policies reproduces their verdicts exactly |

A check that also flags a non-target flaw (for example check 1 firing on P3) is reported but not counted as a failure, since real flaws can overlap.

**Can the validation fail?** Yes, by design. Before step 4, plant one bug in a scratch copy of each check (for example, a threshold off by 10x) and confirm the run catches it, as the BSRS gates did with their five planted bugs.

## Experiment T: do training signals predict robustness?

Experiment T tests the claim that would make Sim2Gate different from Arena: that signals measured during training predict how robust the finished policy is. V1.4 shows ρ predicts how much shaping changes each PPO update; it does not yet show any link to robustness, so this is a new, preregistered experiment, run after the MVP checks pass validation.

**Step 0, an RSL-RL gate.** ppo-shaping-diagnostics is validated against CleanRL and SB3 only. Before any Experiment T data counts, it must pass an RSL-RL acceptance gate built the same way: criteria fixed in advance, comparison against RSL-RL's own advantages and policy gradients, and planted bugs. The gate also records how RSL-RL normalizes advantages, since ρ is defined per normalization group.

**Design:**

| Item | Choice |
| --- | --- |
| Arms | Same-critic BSRS on the Go2 velocity task with η = 0 (raw PPO), 0.5, 1 and 2 |
| Seeds | 1 to 5 per arm, 20 policies in total |
| Logged during training | Per-rollout ρ (RSL-RL normalizes advantages over the whole rollout by default; per-minibatch only if that option is on), regime (cancellation, partial, reversal), zero-tail fraction, and each reward term's share of the return |
| Training-signal summary | Mean 1 − ρ and reversal share over the last 20% of training |
| Outcome | Robustness score: mean success rate across the check 1 sweep grid, using the identical Arena variation draws for every policy |
| Control | Nominal tracking error on PhysX, so robustness is not just "walks better overall" |

**Hypotheses:**

- **T1 (primary):** across the 20 policies, the training-signal summary correlates with the robustness score after controlling for nominal tracking error (partial Spearman, two-sided, 95% bootstrap interval excluding 0). No direction is predicted, because V1.4 found reversal did not hurt outcomes in Mountain Car.
- **T2 (exploratory):** within each η arm, seed-to-seed differences in the signal track differences in robustness, which would show the signal carries information beyond the shaping setting itself.

**Power:** with 20 policies, only large correlations (roughly 0.45 or more) can be detected. Treat this as a pilot: a clear effect justifies a larger study, and a null does not rule out a smaller one.

**If the result is null:** Sim2Gate keeps reward attribution as a diagnostic on mechanism grounds, explaining what shaping did to training, but does not claim it predicts robustness. The null result is reported, as V1.4 reported H1.

## Limits of this validation

Passing this plan shows the checks detect flaws we planted; it does not show they predict real-robot failure. That needs a hardware study with a partner such as Industrial Next's Arena, reporting a sim-vs-real correlation as in [Kadian et al.](https://arxiv.org/abs/1912.06321v2).

- **Planted flaws are cleaner than real ones.** Real policies fail for mixed, subtle reasons.
- **Partly circular for check 1.** P1 and P2 are flawed by the same parameters check 1 sweeps, so detection is expected; the test is mainly that the cliff and flag rule work.
- **Newton is not reality, and not standalone MuJoCo either.** Check 5 shows overfitting to one engine, not readiness for hardware.
- **Small sample:** three seeds per flaw, five held-out baselines. Enough to catch a broken check, not to estimate detection rates precisely.
- **One robot, one task.** Thresholds may not carry over to other robots or tasks.

## Decisions for Vihaan

Each item is a proposal; tick it to approve or leave a comment with a change. Once every item is ticked, the plan is frozen.

- [x] Episode success: 20 s without falling, tracking error at most 1.5x baseline
- [x] Planted flaws P1 to P5 and their config changes
- [x] Seed split: P0 seeds 1 to 3 calibrate, seeds 4 to 8 held out; 3 seeds per flaw
- [x] Check 2 thresholds: 1.5x the calibration maximum; actuator limits absolute at 1% of steps
- [x] Check 1 plausible real ranges and the 80% cliff rule
- [x] Check 5 on Arena's PhysX vs Newton engines, its parity conditions and flag rule (80% success, error ratio 2)
- [x] Acceptance criteria A1 to A6
- [x] Blinding procedure and who holds the key
- [x] Planted-bug test of the checks before the blinded run
- [x] Building checks 1, 2 and 5 as Arena plug-ins, and contributing the physics variations and exploit metrics back to Arena
- [x] Experiment T: arms, seeds, signal summary, outcome and hypotheses T1 and T2
- [x] RSL-RL acceptance gate for ppo-shaping-diagnostics before any Experiment T data counts

Approved by Vihaan on: 10/12/2026 · Plan SHA-256: \_\_\_
