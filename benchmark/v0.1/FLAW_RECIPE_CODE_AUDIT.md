# Flaw recipe implementation audit — 2026-10-27

**Status: source inspection only, no Isaac Lab runtime execution.** Reviewed RootStep branch `benchmark/v0.1-foundation`: `src/sim2gate/mvp/tasks.py`, `src/sim2gate/v2/tasks.py`, `src/sim2gate/v2/phase2_arms.py`, `src/sim2gate/v2/phase1.py`, `src/sim2gate/v2/layer1_limits.py`. Existing variants and historical results are **seen**; never reuse as fresh confirmatory benchmark.

| Recipe | Existing code evidence | Feasibility | Open test before pilot |
|---|---|---|---|
| T1 reduced slip penalty | v2 `phase2_arms.ARMS['R-slip']` scales `feet_slide` by 0.1 or 0.03; `mvp.tasks._p0_env_cfg` explicitly adds `mdp.feet_slide` to RootStep P0 (stock task was different). | **Implemented in custom P0 family** | Confirm term nonzero in saved training config/logs, baseline reward contribution, independent slip manifestation |
| T2 tracking/stability tradeoff | Existing `R-jitter` scales `action_rate_l2`, `R-hop` scales `feet_air_time`; neither directly establishes proposed speed-tracking-vs-stability tradeoff. | **Requires new targeted variant** | Identify exact reward terms, measure actual contributions and qualification |
| D1 narrow friction | `Phi-slip` changes static friction ranges (0.2–0.5, 1.0–1.2); dynamic friction 0.75×static. Historical v1 P1 also narrows friction. | **Implemented historically, adapt** | Distinguish narrow range vs shifted range, predefine held-out friction stress and matched control |
| D2 delay/actuator | `Phi-delay` sets `max_delay_steps=0`; P0 v5b has motor strength 0.8–1.0 in training. | **Delay implemented; motor-strength ablation requires new variant** | Check delay actually varies during clean training and evaluation has meaningful latency |
| S1 simulator exploit | v1 P3 combines low friction and removes slip penalty; v2 `R-slip` separately reduces slip penalty. | **Proxy exists; exploit mechanism NOT established** | Predefine nonphysical contact exploit criterion; distinguish ordinary sliding from simulator exploitation |
| E1 engine dependence | `phase1.measure_commands` evaluates nominal PhysX and Newton; `layer1_limits.clean_report` calls transfer score. | **Measurement implemented, planted cause NOT established** | Match configurations and validate cross-backend outcome reproducibility; do not assert engine dependence as intentional defect |
| N1 naturally occurring | Phase 1 pilot and qualification track failures and failed seeds; v2 Phase 2 has an incomplete seed. | **Prospective protocol required** | No planted cause; blinded detection labels, unknown-cause treatment and no post-hoc selection |
| H0 healthy | Phase 1 `QUALIFY_MIN=0.85` over 256 nominal episodes; training returns and measurements recorded. | **Historical qualification exists** | Benchmark qualification must be specified anew without using Sim2Gate diagnostic thresholds or selecting on its outputs |

## External overlap check (official sources)

- **Arena 0.3:** per-episode variation draws and outcomes, **joint success-conditioned posterior sensitivity** using NPE/MNPE. It is not merely one-factor-at-a-time stress testing. Treat as an explicit baseline; the published Arena example is manipulation-oriented, so Go2 adapter and equivalence are unverified. https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/concepts/concept_sensitivity_analysis.html
- **Arena 0.3:** subtask predicates, distributed evaluation and actionable failure analysis already exist; no novelty claim for locating task-stage failure or perturbation sensitivity. https://github.com/isaac-sim/IsaacLab-Arena/discussions/1245
- **RoboGauge:** Go2 multi-terrain, sim-to-sim policy evaluation with tracking, safety and quality metrics; candidate named comparator for S1/E1 and robustness. Check actual checkpoint compatibility, metric definitions and evaluation budgets. https://robogauge.github.io/ ; https://github.com/wertyuilife2/go2_rl_robotlab

## Decision for the next development-only pilot

**Recommended candidates, not yet approved:** H0 matched control, T1 reduced slip penalty, D1 narrowed friction range, D2 removed delay. These have existing variant mechanisms and measurable independent outcomes. **Defer** T2 until targeted config exists; **defer S1** until a simulator-specific exploit can be established; **defer E1 as planted defect** (retain cross-engine outcome); keep N1 exploratory and separately labeled. Pilot size and exact seeds remain OPEN; no training authorized.

## Required code-level verification before GPU use

1. Resolve actual task registration and save effective `env_cfg` and `agent_cfg` snapshots for each candidate; assert single intended intervention against matched H0.
2. Verify reward term nonzero/changed for T1, friction distribution sampled for D1, and nonzero clean delay with zero intervention delay for D2.
3. Verify train/eval configuration separation, measurement reproducibility and independent reference-grid labeling.
4. Establish matched baseline adapters for Arena-style joint sensitivity and RoboGauge-style scores, or document adapter infeasibility rather than claim a fair head-to-head.
5. Complete CPU synthetic scoring/leakage dry run and decide how to handle failed manipulations before any confirmatory freeze.

**No protocol freeze or new held-out training implied.**
