# Independent custody review packet — Diagnostic Benchmark v0.1

**Status: PREPARATION ONLY.** No real sealed truth key exists in this handoff, no reviewer has attested, and the benchmark protocol is NOT FROZEN. This document is suitable for independent review of the *synthetic* engineering boundary, not for confirmatory evaluation approval.

## Scope and roles

- Custodian: Vijendra Kulkarni. Retains exclusive control of future real labels and custody records.
- Scorer / technical development: separate identity from the custodian. No access to real labels or intervention recipes.
- Independent reviewer: **not yet selected**. Must be independent of scoring and able to inspect access-control evidence and commitment order.
- Review object: blinded scoring boundary and operational handoff controls; **not** a scientific endorsement of benchmark thresholds or performance.

## Reproducible synthetic evidence

- Source: branch `benchmark/v0.1-foundation`, test `benchmark/v0.1/tests/test_os_custody_isolation.py`.
- Passing CI: https://github.com/RootStep/sim2gate/actions/runs/40884596568
- Tested source commit: `4fe14edd45f9aaaeac5e80709a3965d51aa5127b`.
- Test creates a fake answer key in a private directory; unprivileged Linux `nobody` scorer explicitly attempts to read the key and must encounter `PermissionError`. The custodian/root process can read it. Scorer reads public synthetic measurements and writes predictions; custodian verifies digest and synthetic key membership.
- Companion tests: `test_custody_boundary_regressions.py` (truth invariance, metadata injection, tamper/ID rejection and demonstration that editable SHA-256 is not authentication); `test_process_custody_canary.py`; `test_real_production_parity.py` (synthetic inputs invoking real Check 1/2/5).
- **Limits:** the Linux CI test is not the real custody environment; its root-to-nobody boundary is not independent reviewer attestation. The synthetic smoke scorer is not the frozen scientific scorer. No signed external commitment exists.

## Reviewer checklist (complete with evidence; no secrets in repository)

- [ ] Identify independent reviewer, their affiliation and conflict-of-interest statement.
- [ ] Review exact test commit and passing CI logs; independently reproduce synthetic permission denial.
- [ ] Confirm scorer cannot traverse/read private key storage even when given the key path; confirm public inputs remain readable and scorer runs.
- [ ] Verify no intervention labels, recipes, checkpoint identifiers, seed metadata, report paths or private key paths enter the scorer's inputs or outputs.
- [ ] Verify changes to fake truth do not alter predictions, and tampering with predictions or opaque IDs fails verification.
- [ ] Inspect the planned real storage identity and permissions **privately**, without copying secrets, labels or sensitive paths into GitHub or chat.
- [ ] Witness an external, independently write-protected pre-unseal record of public-input digest, prediction digest, scorer commit, configuration digest, UTC time and operator. A SHA-256 inside editable JSON is insufficient.
- [ ] Verify the exact opaque-ID membership and abort-on-mismatch rule.
- [ ] Confirm key access logging, reviewer witnessing, exception handling and incident/abort procedure.
- [ ] Record findings, deviations, remediation and explicit reviewer signoff. No signoff is implied by a passing CI run.

## Required private handoff record (blank template)

| Field | Status |
|---|---|
| Reviewer name and independence/conflicts | PENDING |
| Custodian and scorer runtime identities | PENDING |
| Private key storage access-control proof | PENDING |
| Synthetic production-identity negative-access witness | PENDING |
| Public input commitment and external storage reference | PENDING |
| Prediction commitment and external storage reference | PENDING |
| Scorer commit and configuration digest | PENDING |
| UTC timestamps and operator | PENDING |
| Exact ID membership verification | PENDING |
| Independent reviewer signature/date | PENDING |
| Protocol and scientific freeze approval | NOT APPROVED |

**Stop rule:** no real labels created/shared/unsealed, no held-out training/evaluation, and no declaration of independent custody completion until the scientific protocol is explicitly approved and the private custody review is signed off. The custodian should never post the key or its private storage path in GitHub, CI logs, or chat.
