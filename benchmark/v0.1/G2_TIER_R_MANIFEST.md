# G2 execution manifest — Phase 2 tier R (DRAFT, not approved)

**Status: DRAFT — 2026-10-30.** The machine-readable manifest is [G2_TIER_R_MANIFEST.json](G2_TIER_R_MANIFEST.json). The collector refuses to run unless its `status` is `APPROVED`. Approving means:
- filling in `approval`: who approved, the time, the GPU-hour cap and the cost cap;
- resolving `decisions_open`;
- setting `status: "APPROVED"`;
- committing that version **before** the first run.

Any later change to cells, rules, commands or policies needs a new manifest version.

## What runs

| Item | Value |
|---|---|
| Policies | 19, from the byte-verified records: controls p0v5b_s101–s105; R-slip s301–s305; Φ-slip-mild s306–s310; Φ-delay s311, s312, s314, s315 |
| Common task | `Sim2Gate-Velocity-Flat-Go2-P0-v5b` for every policy; never a training task |
| Environment | Isaac Lab `91a0906`, Isaac Lab-Arena `2cc9e97`, rsl-rl-lib 5.4.1, container `isaac-arena-vscode` on `sim2gate-arena-2` (L40S); play seed 20261031 |
| Fixed conditions | Sim2Gate's nominal profile (randomization and pushes removed, task observation noise on, zero motor delay), plus all terminations except the time limit removed |
| Commands | [configs/tier_r_commands_64.json](configs/tier_r_commands_64.json): 64 commands, forward 0.3–1.0 m/s, lateral ±0.3, yaw ±0.5, seed 20261031. No overlap with Check 2's set |
| Episodes | 128 per policy per cell (2 × 64 commands), so 640 pooled control episodes per cell |
| Runs | 19 policies × 7 cells = 133. Estimated 5–7 GPU-hours; cap still to be set |

| Cell | Static friction | Latency | Role |
|---|---|---|---|
| N080 | 0.80 | — | Nominal reference |
| F040 | 0.40 | — | Low friction, off Check 1's grid. **Primary** for R-slip and Φ-slip-mild |
| F045 | 0.45 | — | Low friction, off Check 1's grid. **Primary** for R-slip and Φ-slip-mild |
| F110 | 1.10 | — | High friction |
| L1 | 0.80 | Observation, 1 step (20 ms) | **Primary** for Φ-delay |
| L2 | 0.80 | Observation, 2 steps (40 ms) | **Primary** for Φ-delay |
| L3 | 0.80 | Observation, 3 steps (60 ms) | Secondary: outside the 0–30 ms plausible envelope; reported separately |

## How truth is decided

- [independent_analysis.py](independent_analysis.py) uses only [independent_truth.py](independent_truth.py). It imports no Sim2Gate code.
- An episode fails on a fall (independent detector) or on tracking loss.
- A policy's truth is "manifested" if, in at least one of its arm's two primary cells, both hold:
  - its failure rate exceeds the pooled controls' with one-sided Fisher exact p below 0.01 / 2 (Bonferroni);
  - the difference is at least 0.10.
- All other cells are reported as cross-symptoms.
- Missing runs stay missing; nothing is imputed.

## Run order once approved

For each policy and cell, inside the container from `/workspaces/isaaclab_arena`:

```bash
/isaac-sim/python.sh v2_runs/phase2_eval_tools/independent_collector.py --manifest v2_runs/phase2_eval_tools/G2_TIER_R_MANIFEST.json \
  --policy <id> --cell <cell> --out v2_runs/phase2_tier_r/<id>__<cell>.jsonl.gz -- \
  --task Sim2Gate-Velocity-Flat-Go2-P0-v5b --checkpoint <manifest checkpoint path> \
  physics=isaacsim_physx env.sim.use_newton_actuators=false --external_callback sim2gate.v2.tasks.register --seed 20261031
```

**First run one smoke test**, one control in N080. Check its header before anything else:
- `checkpoint_matches_manifest: true`;
- `removed_terminations` lists everything but the time limit;
- `contact_material` is 0.8 / 0.6;
- each of the 128 episodes runs the full episode length.

Before the full set:
- run the tier-R Sim2Gate rerun (spec section 6) in the same common task;
- commit its table before any truth analysis.

## Open decisions (also listed in the JSON)

1. **Latency mechanism.** Observation-side control-step latency amends spec section 5 from "values not on Check 1's frozen grid" to "a different delay mechanism". See [INDEPENDENT_TRUTH_G1.md](INDEPENDENT_TRUTH_G1.md).
2. **`truth_rules` values.** They are currently the G1 proposals.
3. **Episodes per cell.** Currently 128.
4. **GPU-hour and cost caps.**
