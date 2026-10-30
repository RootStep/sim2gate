# Diagnostic Benchmark v0.1 — H0/T1/D1/D2 development pilot (review draft)

**Status: DRAFT / DESIGN ONLY — 2026-10-30.** This document authorizes **no GPU training, no held-out generation, no unsealing, and no protocol freeze**. All pilot policies are **seen development data** forever. This pilot does not replace the benchmark protocol in `PROTOCOL.md`.

## 1. Question and scope

Can matched healthy and intentionally modified Go2 locomotion training recipes produce *reproducible, independently measured behavioral differences*, without using the planted intervention as proof that the symptom exists? The purpose is to establish feasibility, estimate variability and compute, and refine a **future** confirmatory design. It is not an estimate of Sim2Gate detection accuracy.

**Protocol discrepancy to resolve before execution:** `PROTOCOL.md` currently proposes **two healthy + two flawed** policies, whereas the H0/T1/D1/D2 shorthand suggests **one healthy + three flawed** policies. This draft treats H0/T1/D1/D2 as four *pilot slots*, **not** an approved cohort assignment. The number of controls and flawed runs remains an explicit approval item. A single control is not enough to estimate between-seed variation.

## 2. Existing evidence and candidate interventions (not new results)

The frozen historical v2 Phase 2 protocol (`docs/v2/PHASE2_PROTOCOL.md`, revision 4) reports:
- **R-slip:** `feet_slide` reward weight ×0.1, prior pilot seeds 201–202 passed.
- **Φ-slip-mild:** training static friction 0.3–0.6 with dynamic friction at 0.75×, prior pilot seeds 219–220 passed.
- **Φ-delay:** no command delay in training (`max_delay=0`), prior pilot seeds 221–222 passed, evaluated as **delay fragility**, not jitter.
- Prior R-jitter and R-hop interventions did **not** pass the earlier pilots and must not be described as reliable flaw generators.

**Proposed candidate mapping for review, not an assignment:**
- H0: unmodified P0 v5b healthy training recipe.
- T1: R-slip candidate (reward-side slip).
- D1: Φ-slip-mild candidate (physics-side slip).
- D2: Φ-delay candidate (delay fragility).

This gives two mechanisms for slip plus a distinct delay symptom; it does **not** establish the benchmark's broader five/six-category taxonomy. These candidates have already been explored in v2. Any new pilot must use new run IDs/seeds and must be excluded from the future held-out cohort.

## 3. Pre-execution approval table

| Decision | Proposed approach | Status |
|---|---|---|
| Cohort composition | Choose between 2 healthy + 2 flawed (existing benchmark protocol) and 1 healthy + 3 flawed (H0/T1/D1/D2 shorthand); consider additional controls for variability | OPEN |
| Candidate interventions | H0/T1/D1/D2 mapping above; confirm no reuse of v2 pilot/study seeds or checkpoint lineage | OPEN |
| Training recipe | Pin Isaac Lab/RSL-RL version, source commit, P0 v5b config, reward/physics diff and training duration | OPEN |
| Fresh seed allocation | Explicit nonoverlapping seeds, registered against all 97 seen exclusions and historical run families | OPEN |
| Evaluation conditions | Nominal PhysX, Check 1 disturbance grid, Check 2 measurement conditions, Check 5 PhysX/Newton; pin commands, versions and per-condition budgets | OPEN |
| Applicability | Check calibration validity before reusing any historical v2 thresholds; otherwise use development-only thresholds and record provenance | OPEN |
| GPU budget | Measure current machine throughput and approve hard maximum GPU-hours and cloud cost before starting | OPEN |
| Stop/retry rules | Predeclare handling of training crashes, absent checkpoint, incomplete evaluations, policy collapse and reruns | OPEN |
| Artifact custody | Hash all configs, scripts, run IDs, checkpoints, logs, calibration and raw outputs; keep private labels and secrets out of public feature exports | OPEN |

## 4. Proposed measurements and manipulation checks

Record **all assigned runs**, including failed and nonmanifesting ones. Never silently replace a weak flawed policy. Evaluate each against matched healthy controls using the same measurement code and budget.

| Candidate | Target evidence to measure | Historical v2 context (not automatically transferable) |
|---|---|---|
| H0 | Nominal success, tracking error, slip p95, Check 1 delay cliff, Check 2 metrics, Check 5 engine transfer | Healthy must be defined by tested envelope, not nominal success alone |
| T1 R-slip | Elevated foot-slip p95 with nominal viability | Historical v2 pattern: slip p95 >0.1185 m/s and nominal success ≥0.85 |
| D1 Φ-slip-mild | Elevated foot-slip p95 with nominal viability | Same historical pattern; intervention family differs |
| D2 Φ-delay | Check 1 `action_latency_ms` flagged, without slip manifestation | Historical v2 pattern: delay flag and slip p95 ≤0.1185 m/s |

The historical v2 threshold 0.1185 m/s and success cutoff 0.85 are **context only**. Before this pilot, document whether they apply to its exact policy, calibration, command set, physics, and measurement code. If not, choose and label **new development-only** criteria before observing pilot outcomes.

**Suggested comparison outputs:** per-run target symptom, magnitude versus each matched healthy control, nominal viability, non-target side effects, invalid/missing check counts, and confidence/variability across seeds. If only one healthy control is run, report uncertainty as **not estimable** rather than implying replication.

## 5. Proposed run workflow (blocked until approval)

1. Freeze a **pilot-only** plan snapshot (not `FREEZE.md`), source commit, seed allocation, config diffs, environment versions, cost cap and run manifest.
2. Run CPU-only schema, leak, custody and production-check parity tests.
3. Train and evaluate development runs under identical logging, command and disturbance conditions; save all crashes and failed checkpoints.
4. Perform blind-to-recipe behavioral measurement and independent manipulation checks; distinguish planted intervention, manifested symptom and hypothesized mechanism.
5. Report complete results, negative outcomes, invalid applicability, run costs and confidence limits.
6. Decide whether to redesign recipes, add healthy replication, or prepare a separate full benchmark freeze proposal.

## 6. Required artifact manifest

Per run: immutable run ID, parent/source commit, recipe and diff hashes, seed, trainer versions, GPU model, runtime/cost, full training exit status, checkpoint SHA-256, selected checkpoint rule, raw evaluation manifests, Check 1/2/5 reports and calibration provenance, independent symptom report, errors/retries, and a public sanitized diagnostic export. Keep any held-out key or private custody location **outside** this pilot package.

## 7. Explicit non-goals and gates

- No confirmatory AUROC/attribution/localization claims from these four development slots.
- No change to frozen historical v2 experiments.
- No benchmark `FREEZE.md` approval, held-out generation, or one-shot evaluation.
- No assumption that a reward/physics change is itself a behavioral flaw.
- No transfer-to-hardware claim from PhysX/Newton comparisons.
- **Go/no-go for pilot execution:** all nine decisions in section 3 resolved, independent seed-lineage check, production-check applicability verified, GPU budget approved, and explicit manager authorization.
- **Go/no-go for future benchmark freeze:** reproducible manifestation evidence across independent seeds, defined taxonomy, valid calibration, independent dense-grid truth, baseline implementations, power/precision justification, custody review, and signed freeze record.

## 8. Decisions requested from RootStep

**First decision:** approve a pilot design with **two healthy + two flawed** policies (consistent with `PROTOCOL.md`), or revise that protocol explicitly to **one healthy + three flawed** (H0/T1/D1/D2). No GPU resources will be used until the full plan is approved.

**Second decision (later):** approve the exact fresh seed set, recipe hashes, thresholds and GPU-hours/cost cap once the execution manifest is prepared.
