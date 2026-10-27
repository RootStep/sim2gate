# Benchmark v0.1 — repository gap analysis

**Status:** initial source inspection, not an executed end-to-end validation.

| Requirement | Existing foundation | Work needed |
|---|---|---|
| Go2 training, healthy and planted recipes | `src/sim2gate/mvp/baselines.py`, `src/sim2gate/v2/phase1.py`, `src/sim2gate/v2/phase2_arms.py` | New independent cohort recipes and manipulation checks |
| Stress tests, slip, engine comparison | `src/sim2gate/checks/` | Frozen dense failure-condition reference grid |
| Training diagnostics | `src/sim2gate/training/`, `src/sim2gate/v2/phase2_export.py` | Comparable baseline features and missingness policy |
| Blinding | `src/sim2gate/mvp/blinded.py`, `src/sim2gate/v2/phase2_study.py` | Separate custodian, leak checks and benchmark-specific lock |
| Thresholds | `src/sim2gate/v2/layer1_limits.py` | Verify applicability and calibrate benchmark FAR |
| Classification | `src/sim2gate/v2/phase2_score.py` | General detection, attribution, location scoring and policy-level uncertainty |
| Baselines | Nominal/sweep/training feature code exists | Implement unified adapters and documented Arena-style sensitivity comparator |
| Reproduction | Existing tests, protocols and hashes | Benchmark-specific manifests, CI and clean-environment rerun |

## Scientific constraints
The October 26 v2 Phase 2 conclusion reports **19/19** intervention classifications for training diagnostics versus **17/19** behaviour only among completed runs, but the preregistered improvement criterion **C2 was not met**; adding rho did **not** meet **C3**. One assigned run failed to complete and the original 20-row analysis departed from the frozen exclusion rule. Do not advertise a validated diagnostic advantage on that evidence.

Prior 23-policy blinded data are **seen** and cannot be counted as a fresh confirmatory set. Phase 3 fix verification is separate.

## Immediate priorities
**P0:** Evidence inventory, taxonomy and primary metric decisions, independent ground truth, threshold provenance, freeze protocol.

**P1:** Versioned prediction/truth schemas, baseline adapters, scoring tests, leak tests and CPU dry run.

**P2:** Small fresh GPU pilot to estimate feasibility and budget, then approval of final held-out size.

## Audit limitations
Repository structure and selected code/protocols were inspected through connected GitHub. Tests and GPU experiments were not executed; full evidence bundles, other branches and local uncommitted code were not verified.
