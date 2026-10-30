# G2 execution manifest — Phase 2 tier R (DRAFT revision 2, not approved)

**Status: DRAFT, revision 2 — 2026-10-30.** The machine-readable manifest is [G2_TIER_R_MANIFEST.json](G2_TIER_R_MANIFEST.json). This revision answers the reviewer's G2 decision; the item-by-item response is in [G2_REVISION2_RESPONSE.md](G2_REVISION2_RESPONSE.md).

**Review status:** G1 provisional. G2 provisionally accepted. The S0 smoke test is pending explicit manager authorization, and S1 and S2 are not authorized. **Nothing is authorized to run.**

The reviewer's additional finding (revision 2b) is answered in [G2_REVISION2_RESPONSE.md](G2_REVISION2_RESPONSE.md#revision-2b-step-synchronization-and-effective-friction).

The collector refuses to run unless both of these hold:
- the manifest's `status` is `APPROVED`;
- the stage named on the command line (`--stage`) is `APPROVED` and lists that exact policy and cell.

Approving the manifest does not approve any stage. Each stage needs its own approval, recorded in the JSON (`approved_by`, `approved_utc`) and committed before its first run. Any change to cells, rules, commands or policies after S1 starts needs a new manifest revision.

## What runs

| Item | Value |
|---|---|
| Policies | 19, from the byte-verified records: controls p0v5b_s101–s105; R-slip s301–s305; Φ-slip-mild s306–s310; Φ-delay s311, s312, s314, s315 |
| Common task | `Sim2Gate-Velocity-Flat-Go2-P0-v5b` for every policy; never a training task |
| Environment | Isaac Lab `91a0906`, Isaac Lab-Arena `2cc9e97`, rsl-rl-lib 5.4.1, container `isaac-arena-vscode` on `sim2gate-arena-2` (L40S); play seed 20261031 |
| Fixed conditions | Sim2Gate's nominal profile (randomization and pushes removed, task observation noise on, zero motor delay), with every termination except the time limit removed |
| Commands | [configs/tier_r_commands_64.json](configs/tier_r_commands_64.json): 64 commands, forward 0.3–1.0 m/s, lateral ±0.3, yaw ±0.5, seed 20261031; no overlap with Check 2's set |
| Episodes | 128 per policy per cell (2 × 64 commands), 1000 control steps (20 s) each; 640 pooled control episodes per cell |
| Runs | 19 policies × 8 cells = 152, plus 1 smoke run = 153 |

| Cell | Static friction | Delay | Role |
|---|---|---|---|
| N080 | 0.80 | — | Nominal reference (secondary) |
| F040 | 0.40 | — | **Primary** for R-slip and Φ-slip-mild; off Check 1's friction grid |
| F045 | 0.45 | — | **Primary** for R-slip and Φ-slip-mild; off Check 1's friction grid |
| F110 | 1.10 | — | High friction (secondary) |
| A1 | 0.80 | Action/command, 1 control step (20 ms) | **Primary** for Φ-delay |
| A2 | 0.80 | Action/command, 2 control steps (40 ms) | **Primary** for Φ-delay |
| O1 | 0.80 | Observation, 1 control step (20 ms) | Secondary generalization test; never decides truth |
| O2 | 0.80 | Observation, 2 control steps (40 ms) | Secondary generalization test; never decides truth |

The 60 ms observation cell (L3) of revision 1 is dropped. It was outside the plausible 0–30 ms envelope and outside Check 1's 0–45 ms grid.

### Why action delay is primary

Φ-delay's planted flaw is that it was trained **without actuator command delay**. Every other arm trained on P0 v5b physics, which includes 0–30 ms command delay. The cell that tests whether that flaw shows up as failures must therefore delay the **command** channel.

The A cells do this. The collector holds the policy's action for 1 or 2 whole control steps before it reaches the action term.

**Separation from Check 1.** The code path is different:
- Check 1 delays the joint target inside the swept DC motor model, in 5 ms physics steps.
- Here the delay is a buffer at the policy interface. The run asserts at runtime that the motor model's lag is zero, so the two never combine.

Outcome labels come only from the independent truth library, with a different command set.

**Limitation (stated, not hidden).** At 20 and 40 ms, the physical perturbation is close to Check 1's grid points of the same values. A1 and A2 are an independent implementation and judgment of the same kind of perturbation. They are not an independent kind of perturbation. Agreement between Check 1 and the A cells on Φ-delay is therefore partly expected by construction, and the report must say so.

The observation cells O1 and O2 delay a different channel. They are reported as a secondary generalization test with no truth weight.

## How truth is decided

[independent_analysis.py](independent_analysis.py) uses only [independent_truth.py](independent_truth.py) and imports no Sim2Gate code. An episode fails on a fall (independent detector) or on tracking loss.

A policy's truth is "manifested" if, in at least one of its arm's two primary cells, both of these hold:
- its failure rate exceeds the pooled controls' with one-sided Fisher exact p below 0.01 / 2 (Bonferroni);
- the difference is at least 0.10.

All other cells are reported as cross-symptoms.

**Thresholds are provisional.** `truth_rules` may be used for the S0 smoke test only. They must be confirmed, or changed in a new revision, before S1 is approved.

### Run validity

A run is **invalid** if `read_run` reports any of these problems:
- the header is not marked completed;
- the footer is missing or does not match the records;
- the run used a different manifest, checkpoint digest, cell definition, command set or task;
- `num_envs` is wrong;
- runtime verification is missing or failed;
- any episode ended before the time limit;
- any episode is not exactly the episode length;
- any episode fails the schema.

An invalid run is excluded everywhere, including from the pooled controls.

Invalid and missing runs are counted separately and never imputed.

The **required** set is:
- each non-control policy in its arm's primary cells;
- each control in every primary cell.

If any required run is missing or invalid:
- the status is `INCOMPLETE`;
- no policy gets a truth label;
- the analysis exits 2.

## Staged execution and caps

All caps are instance wall-clock hours, from the start of a stage's first run (container setup included) to the end of its last run. Costs use the **Brev billed rate of $3.50/h**, stated by the custodian on 2026-10-30. For comparison, the AWS on-demand list price for g6e.2xlarge in us-east-1 is about $2.24/h.

| Stage | Runs | Estimate | GPU-hour cap | Cost cap at $3.50/h | Gate to start |
|---|---|---|---|---|---|
| S0 smoke | 1: p0v5b_s101 × N080, written to a separate directory, never analysed | under 0.1 h | 0.5 | $1.75 | Manifest and S0 approved |
| S1 pilot | 40: controls s101 and s102, plus s301, s306 and s311, × 8 cells | 2.0 h at 3 min/run | 3.0 | $10.50 | S0 exit criteria met; thresholds confirmed; cap re-derived from S0's measured time |
| S2 | 112: remaining 14 policies × 8 cells | 5.6 h at 3 min/run | 7.5 | $26.25 | S1 runs valid; cap re-derived from S1's times |
| **Total** | **153** | | **11.0** | **$38.50** ($24.66 at AWS list) | |

The 3 min/run figure is an assumption. Each run is one Isaac Sim launch plus 1000 steps × 128 environments. For comparison, Phase 2 measurement blocks of several launches took 4–10 min per robot (`state.json`). S0 measures the real figure.

Stop rules:
- A stage stops when its cumulative hours reach its cap.
- A cap is never raised within a stage; a higher cap needs a new approval.
- An invalid or crashed run is retried once from scratch, within the cap.
- S1 is reviewed for validity, timing and cost only. Outcome analysis waits until S2 is complete.

**S0 exit criteria.** S0 exists to resolve the runtime uncertainties before any scientific evaluation. Every criterion is a field in the run header:

| Evidence | Required value |
|---|---|
| Completion | exit 0; `completed: true`; footer verifies; `read_run` reports no problems |
| 1000 consecutive post-physics samples | `sync_checks.samples_checked` = 128,000, with `capture_length_not_consecutive` and `steps_without_physics_advance` = 0; episode lengths min = max = 1000 |
| First recorded state synchronized | `reset_episode_length_max` = 0; the first capture is at episode length 1; `command_mismatch` = 0 at every step |
| Final sample is the terminal transition | `termination_flag_mismatch` = 0; `terminal_steps` = [1000]; `unexpected_early_dones` = 0 |
| No reset state in the trajectory | `done_without_reset_after_capture` = 0; `state_changed_after_capture` = 0; `reset_moved_robot` = 128 |
| Configuration | `runtime_checks.ok`: terminations `['time_out']`; no removed event active; motor lag 0, strength 1.0 |
| Effective friction | `runtime_checks.contact`: every ground and robot physics material with its combine mode; `effective_static` = [0.8] and `effective_dynamic` = [0.6]; no shared or unbound material |
| Cost | wall time per run recorded |

If S0 fails, no scientific run takes place. The output (exit 3, sync failure) or `<out>.diagnostic.json` (runtime-check failure) is reviewed, the collector is revised, and S0 is re-authorized.

## Power

The table below is for 128 policy episodes vs 640 pooled control episodes, one-sided Fisher at α = 0.005 plus the 0.10 floor. It was computed exactly.

| Control failure rate | +0.10 | +0.15 | +0.20 | +0.30 |
|---|---|---|---|---|
| 0.00 | 0.52 | 0.96 | 1.00 | 1.00 |
| 0.02 | 0.50 | 0.94 | 1.00 | 1.00 |
| 0.05 | 0.50 | 0.92 | 1.00 | 1.00 |
| 0.10 | 0.50 | 0.90 | 0.99 | 1.00 |

At 64 episodes, power at +0.15 falls to 0.62–0.86.

With 128 episodes:
- A true excess of 0.15 or more is detected with power of at least 0.90.
- An excess near the 0.10 floor is detected about half the time.

So "not manifested" means "no excess of about 0.15 or more detected", not "no flaw".

Doubling to 256 episodes would give at least 0.8 power at +0.10, and would double the cost of S1 and S2. This is an open decision.

## Run command (only once the stage is approved)

Inside the container, from `/workspaces/isaaclab_arena`:

```bash
/isaac-sim/python.sh v2_runs/phase2_eval_tools/independent_collector.py --manifest v2_runs/phase2_eval_tools/G2_TIER_R_MANIFEST.json \
  --stage <S0|S1|S2> --policy <id> --cell <cell> --out <stage output_dir>/<id>__<cell>.jsonl.gz -- \
  --task Sim2Gate-Velocity-Flat-Go2-P0-v5b --checkpoint <manifest checkpoint path> \
  physics=isaacsim_physx env.sim.use_newton_actuators=false --external_callback sim2gate.v2.tasks.register --seed 20261031
```

Before any truth analysis, run the tier-R Sim2Gate rerun (spec section 6) in the same common task and commit its table.

## Open decisions (also in the JSON)

1. Approval of S0. The Brev rate is confirmed at $3.50/h.
2. `truth_rules` values: provisional until they are confirmed before S1.
3. Episodes per cell: 128, or 256 (see Power).
