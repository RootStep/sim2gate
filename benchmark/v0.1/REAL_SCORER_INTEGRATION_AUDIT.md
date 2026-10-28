# Real Sim2Gate method integration audit — v0.1 Gate 0

**Status: integration NOT complete.** No held-out evaluation or protocol freeze authorized.

## Existing production logic inspected
- `src/sim2gate/checks/robustness.py`: Check 1 robustness sweep, 10 grid values per disturbance, 64 episodes per value, first success-rate cliff below 80%. Requires condition-wise episode data and the matching calibration reference; a scalar `latency_cliff_index` alone does not reconstruct the check.
- `src/sim2gate/checks/exploits_score.py`: Check 2 calibrated raw-metric flags with applicability checks for checkpoint, command-set hash, physics path, sample coverage, measurement settings and thresholds. Requires a full report and calibrated thresholds. Its output includes checkpoint hashes, so **do not pass the raw report directly into a blinded scorer**.
- `src/sim2gate/v2/phase2_score.py`: historical Phase 2 supervised cross-validation. Its stage 1 explicitly loads a label key to create folds and fit models; it **must not** be reused as the blind confirmatory scorer.

## Blocking interface mismatch
The current `public_feature_boundary.py` admits eight aggregate numbers only. It cannot reproduce existing Check 1 or Check 2 applicability and scoring from the production functions without additional *approved public* evidence. Substituting arbitrary weighted sums would silently create a different diagnostic method. The existing `blinded_scorer_smoke.py` therefore remains a synthetic smoke fixture only.

## Required next engineering decision
Build a two-stage **measurement/export adapter** which computes existing production checks in a trusted evaluation environment, strips checkpoint identifiers, seed, recipe, labels and raw file paths, then emits an approved versioned *check-result* schema (flags, missing/applicability status, disturbance-cell outputs, and calibration provenance hash). The blind scorer may consume only this sanitized schema and a precommitted method configuration. Confirm that the adapter never consults the held-out intervention key.

Before implementing, define whether calibrated threshold provenance hashes may reveal training family or policy identity, and specify which fields are public. Do not reuse v1/v2 calibration thresholds without applicability review. Freeze calibration on development data only.

## Governance
Vijendra Kulkarni: designated parent/company-manager held-out label custodian. Vihaan Kulkarni: technical lead, no sealed key access while developing/tuning scorer. Custody location and handoff not yet verified.

**Next acceptance test:** compare sanitized Check 1/2 outputs to original production scoring on synthetic reports, reject injected checkpoint metadata at the blind interface, verify score invariance under changes to sealed labels, and run separate-process key isolation.
