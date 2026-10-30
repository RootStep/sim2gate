# Gate 0 engineering closure evidence — synthetic-only (2026-10-28)

**Scope:** No training, no protocol freeze, no sealed benchmark key, no held-out scoring. Historical recovery backlog deferred in HISTORICAL_CHECKPOINT_RECOVERY_BACKLOG.md.

## Implemented
- public_feature_boundary.py strictly whitelists public numeric measurements and opaque IDs; unexpected fields rejected.
- commit_public_inputs.py creates deterministic public-input SHA-256.
- blinded_scorer_smoke.py is a **synthetic-only placeholder diagnostic** using explicitly illustrative fixed coefficients; it is **not** the real Sim2Gate benchmark method and cannot substantiate detection claims.
- custodian_verify.py independently checks prediction commitment and exact key-ID membership before a synthetic unseal report. It is not a production key-management system.
- tests/test_end_to_end_smoke.py exercises synthetic separated scoring, missing features, planted metadata, commitment tampering and mismatched truth IDs.
- Dedicated GitHub Actions workflow runs synthetic checks and exclusion validation.

## Acceptance boundaries / outstanding
1. Verify latest CI success; preserve run URL and commit SHA.
2. Replace smoke scoring formula with a **reviewed, versioned real Sim2Gate method adapter**, with method inputs and scoring frozen before confirmatory use. Do not present smoke score as Sim2Gate performance.
3. Prove actual runtime separation (distinct credentials/process/workspace and no scorer access to key), not merely separate Python modules.
4. Name the Parent/RootStep company-manager custodian, establish private custody storage and access log, and obtain pre-unseal signature/timestamp.
5. Resolve protocol taxonomy, intervention/manipulation checks, fair baselines, dense-grid truth, calibration and uncertainty rules with independent review.
6. Confirm historical family-level exclusion and ledger exceptions; deferred checkpoint recovery does not permit overlap.
7. Only then consider Gate 0 signoff and separately authorized protocol freeze / pilot.

**Status:** Engineering scaffold implemented; Gate 0 still PARTIAL/BLOCKED. No scientific claim or operational custody certification.
