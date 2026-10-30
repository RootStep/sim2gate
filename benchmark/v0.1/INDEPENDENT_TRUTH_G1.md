# Gate G1 — independent behavioural-truth code (Phase 2 tier R)

**Status: G1 CODE AND UNIT TESTS COMPLETE (CPU) — 2026-10-30.** Implements spec section 5 ([HISTORICAL_PHASE2_INDEPENDENT_EVAL_SPEC.md](HISTORICAL_PHASE2_INDEPENDENT_EVAL_SPEC.md)). It authorizes no trajectory, GPU run or rule freeze: every numeric value below is a **proposal for the G2 manifest**.

## What exists

- [`independent_truth.py`](independent_truth.py) is pure Python standard library. Its functions:
  - strict episode-schema validation;
  - its own fall detector (base height or tilt, sustained), independent of the environment's termination flag;
  - velocity-tracking error after a warm-up, stopped at a fall;
  - a consequence-based episode outcome (fall or tracking loss);
  - stance-phase foot path length and slip ratio;
  - a one-sided Fisher exact test and a matched-control manifestation rule;
  - an observation/action latency buffer.
- [`tests/test_independent_truth.py`](tests/test_independent_truth.py) holds 26 unit tests on synthetic trajectories:
  - stationary vs sliding feet with known path length;
  - swing motion excluded, and touchdown and lift-off trimmed;
  - brief dips vs sustained falls, tilt falls, warm-up and post-fall exclusion;
  - Fisher exact against textbook values (17/70 and 1/252);
  - latency ordering;
  - schema rejection;
  - **independence**: the AST shows the module imports only `math`, and no Sim2Gate module name, limit file or calibrated threshold value appears in the source.

## How it differs from Sim2Gate's checks

| | Sim2Gate | Independent truth |
|---|---|---|
| Slip | Check 2: 95th percentile of instantaneous horizontal foot speed at every physics update in contact | Foot **path length per stance phase**, at the control rate, with touchdown and lift-off trimmed. Secondary only |
| Primary truth | — | **Consequence:** fall (own detector) or velocity-tracking loss, decided per policy against matched healthy controls under identical conditions |
| Latency | Check 1: actuator command delay inside the motor model, 5 ms physics steps, grid 0–45 ms | Delay injected at the policy interface by whole 20 ms control steps (observation or action side) |
| Decision | Fixed calibrated limits | Fisher exact p < alpha **and** a minimum failure-rate difference against controls |

## Decisions the G2 manifest must make

1. **Latency values.** At 5 ms physics and 20 ms control steps, no delay inside 0–30 ms is off Check 1's grid except through Check 1's own mechanism. Options:
   - **(a)** Observation latency of 1 and 2 control steps (20 and 40 ms). Numerically these are grid points, but the mechanism and resolution differ. This needs the spec's "values not on Check 1's frozen grid" amended to "a different delay mechanism".
   - **(b)** 3 control steps (60 ms). Numerically off the grid, but outside the 0–30 ms plausible envelope.

   Proposal: (a) as primary, (b) reported separately.
2. **Friction cells for the slip arms.** Check 1's v2 friction grid is 0.8, 0.725, …, 0.35, 0.275, … Proposal: static friction 0.40 and 0.45 (off the grid, inside the plausible 0.35–1.2), **plus nominal 0.8 and a high cell of 1.1**. All use dynamic friction at 0.75× and the same command set, which differs from Check 2's fixed set.
3. **The numeric rules** in `DEFAULT_RULES`: contact force 10 N, minimum stance 3 steps, trim 1, fall height 0.20 m, tilt 60°, persistence 5 steps, warm-up 50 steps, tracking loss 0.25 m/s, alpha 0.01, minimum rate difference 0.10. Also episodes per cell, command trajectories and seeds.

## Points that could change the result

- **R-slip may not show a consequence at nominal friction.** Historically it slid about 24× the Check 2 limit while succeeding 98–100% of the time. With consequence-based truth, R-slip "manifests" only if its slip causes falls or tracking loss in some cell. That is the substance of the benchmark question, not a defect in the test. Report it as found, including a null result.
- **Φ-slip-mild was trained on low friction (0.3–0.6).** It may do as well as or better than controls on the low-friction cells. Its historical Check 2 slip was measured on that same low friction, so its "flaw" may not exist as a consequence at nominal or high friction either. Every friction cell is reported, so this stays visible.
- **The independent slip magnitude** is a different formulation of the same physical quantity Check 2 measures. That reduces circularity but doesn't remove it, which is why it is secondary.

## Follow-on pieces (written 2026-10-30, not yet run)

- [`independent_collector.py`](independent_collector.py) is the Isaac Lab trajectory collector. It reuses Sim2Gate's environment plumbing (`apply_profile`, command pinning, `install`) and no scoring code. It sets per-cell friction, removes every termination except the time limit, injects latency, and writes `independent-truth-episode-v1` records. It refuses to run unless the G2 manifest is `APPROVED`. Its Isaac-facing parts are untested until a GPU smoke run.
- [`independent_analysis.py`](independent_analysis.py) labels each policy offline from the raw runs, using only `independent_truth.py`.
- [`G2_TIER_R_MANIFEST.json`](G2_TIER_R_MANIFEST.json) and its summary [`G2_TIER_R_MANIFEST.md`](G2_TIER_R_MANIFEST.md) are a DRAFT awaiting decisions and approval.
- [`tests/test_independent_pipeline.py`](tests/test_independent_pipeline.py) holds 12 tests:
  - the episode log;
  - the collector's refusal rules;
  - collector and analysis independence;
  - end-to-end analysis on synthetic runs: manifestation, cross-symptoms, missing runs, wrong manifest;
  - manifest consistency with the byte-verified records.
