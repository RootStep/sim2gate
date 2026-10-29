# Synthetic custody handoff protocol — RootStep v0.1

**Custodian:** Vijendra Kulkarni, parent / RootStep company manager.
**Technical lead:** Vihaan Kulkarni.
**Status:** Procedure documented; no real sealed answer key has been handed off or unsealed. No training or freeze authorized.

## Development-only handoff
1. A synthetic key holder generates opaque IDs and a fake truth key, stored outside the scorer working directory. Do not commit real keys to GitHub.
2. The scorer operator receives only allowlisted, synthetic public features. Any production Check 2 reports must first pass the evaluation-side sanitized adapter. The operator cannot receive report paths, checkpoint hashes, intervention recipes, seeds or key paths.
3. Run the scorer in a separate process with a restricted working directory and credentials. Verify its environment cannot read the key; separate Python modules alone are not access isolation.
4. Commit the input SHA-256, prediction SHA-256, scorer source commit, configuration digest, UTC timestamp and operator before unsealing. The current development-only code commits input and predictions but does not yet implement signed timestamp or configuration attestation.
5. Custodian verifies the commitment and exact opaque-ID membership before releasing the synthetic truth. Any mismatch aborts evaluation.
6. Log who accessed the key, when, why, commitment IDs, deviations and whether a key was unsealed. Keep the key in parent-controlled private storage.
7. For confirmatory use, an independent reviewer verifies the sealed-key generation and custody procedure, not merely the parent role.

## Synthetic dry-run acceptance criteria
- Clean metadata-only scorer process cannot read key or its location.
- Changing sealed synthetic truth without changing public features cannot change predictions.
- Altering predictions after commitment fails verification.
- Missing/invalid measurements are explicitly unscored, never assumed healthy.
- Only opaque IDs and approved numeric/boolean outputs leave the evaluation-side adapter.
- No scorer threshold, calibration or method revision occurs after held-out key exposure.

## Limitations
The synthetic smoke scorer is not the frozen scientific method. Check 1/2/5 adapters and unmocked synthetic production-scored parity tests are implemented and passed CI on commit 0cd4ce71. Synthetic custody regressions now test label invariance, metadata injection, tampering, exact key membership, and the limitation that a digest can be recomputed by a writer. Synthetic Linux OS-identity isolation passed in GitHub Actions on commit 4fe14edd (Gate 0 run 40884596568); real operational identity isolation, independently witnessed handoff, calibration applicability review and protocol approval remain outstanding. The documented procedure is **not** evidence of a completed operational handoff.

## Required real operational access-isolation gate (NOT YET EXECUTED)

The existing separate-process test uses the same OS identity. It is not proof of key inaccessibility. Before a real sealed-key handoff:

1. The custodian must create and retain the real key in a private account/storage location inaccessible to the scorer operator and its runtime identity. Do not share the key or storage path in GitHub, logs or chat.
2. The scorer must run under a different, least-privileged identity or an independently enforced container/credential boundary. Record the identity and permissions in a private custody audit; avoid putting key paths in scorer logs.
3. A custodian-supervised **synthetic** negative-access test must demonstrate that the scorer identity cannot read the synthetic private key (permission denied), while the custodian can. Mere absence of the path from arguments is insufficient.
4. Independently store the input and prediction digests, scorer commit, configuration digest and timestamp in a write-protected or externally witnessed record **before** any key access. A self-consistent digest inside an editable prediction JSON is not authentication.
5. Have an independent reviewer witness the synthetic test and attest to separation and commitment order. Abort on any access or digest discrepancy.

**Current result:** synthetic Linux OS credential isolation passed (commit 4fe14edd; Gate 0 run 40884596568), along with logical/data boundary regressions. This is an automated development test, not a custodian-supervised test of the future production identities. Real operational credential isolation, protected external commitment and independent witnessed handoff remain pending. No claim of complete blindness or readiness for confirmatory evaluation is made.

See `INDEPENDENT_CUSTODY_REVIEW_PACKET.md` for the reviewer checklist and evidence requirements. No real answer key is needed for the synthetic review.
