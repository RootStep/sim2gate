# Diagnostic Benchmark v0.1 — decisions required before freeze

**Status: PROPOSED / NOT APPROVED (Oct 27, 2026).** This is a decision log, not a protocol amendment or authorization to train. See [PROTOCOL.md](PROTOCOL.md) and [GATE0_VERIFICATION_UPDATE.md](GATE0_VERIFICATION_UPDATE.md).

## Decisions to resolve

| ID | Question | Proposed decision for review | Required validation |
|---|---|---|---|
| D1 | Taxonomy says six but names five | Define five categories: training, dynamics, simulator exploits, engine dependence, naturally occurring; healthy is a separate control class. Do not force naturally occurring cases into a known-cause attribution score. | Named operational recipes, label hierarchy and eligibility |
| D2 | What does planted flaw mean? | Keep intervention label separate from measured manifestation and mechanism attribution; failed manipulation checks remain reported, not silently removed. | Independent manipulation-check thresholds and adjudication |
| D3 | What is a healthy control? | No planted intervention plus pre-specified nominal and stress qualification; report nonqualifiers and selection effects. Do not equate 'healthy' with infallible. | Predefine qualification using development data only |
| D4 | Primary diagnostic claim | Detect flawed-vs-control at a calibration-set threshold; use policy-level sensitivity at target 10% false-positive rate as proposed primary, AUROC secondary. Attribute/Locate are separate outcomes. | Fix calibration unit, tie/missing rules, confidence intervals and success criteria |
| D5 | Baseline fairness | Nominal success, robustness aggregate, individual raw signals, and documented Arena-style sensitivity analogue with matched evaluation information/budgets where feasible. | Define exact feature inputs, access, compute budgets, and N/A handling |
| D6 | Location ground truth | Independently measure disturbance-cell failures on a sealed dense reference grid not supplied to predictors. | Disturbance grid, replicate count, cell failure rule, measurement error |
| D7 | Splits and leakage | Historical v1/v2 policies are seen; use new development/calibration and untouched confirmatory training runs; lock recipes, seed assignment and answer-key custody before generation. | Explicit run-family separation, metadata audit and access log |
| D8 | Cohort size | Treat 40 flawed + 40 healthy as conditional, not committed. | GPU pilot runtime/qualification rate and policy-level precision/power calculation |
| D9 | Failure/retry rules | Preserve all assigned attempts and failed checkpoints; distinguish training failure, failed manipulation, missing evaluation, and excluded data in analysis. | Predeclare denominator and intention-to-test analyses |
| D10 | Reproducibility gate | CPU synthetic scoring/leakage dry run, then small development GPU pilot, then signed/hash-committed protocol and one-shot confirmatory run. | Tests, artifacts, custodian, environment versions |

## Immediate execution order
1. Finish Gate 0's complete seen-policy ledger and manifest checks; historical Phase 1-to-Phase 2 controls 101–105 now have byte-matching SHA-256 evidence.
2. Resolve D1–D6 in writing with an independent reviewer; specifically prevent circular definition of truth using Sim2Gate's own thresholds.
3. Implement versioned prediction/truth schema and baseline adapters, and run CPU dry-run with missing data and leakage adversaries.
4. Run a small **development-only** GPU feasibility pilot, estimate sample size and cost, then resolve D7–D10.
5. Only then approve and freeze `PROTOCOL.md` and `FREEZE.md` with commit and SHA-256 evidence.

**Nonclaims:** No new benchmark policies trained; no held-out performance established; no hardware sim-to-real validation claimed; Phase 3 fix verification remains outside v0.1.
