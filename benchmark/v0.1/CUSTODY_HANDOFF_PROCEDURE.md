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
The synthetic smoke scorer is not the frozen scientific method. Check 2 adapter is implemented, but full Check 1/5 production adapter, runtime key isolation, independently witnessed handoff, calibration applicability review and protocol approval are still outstanding. The documented procedure is **not** evidence of a completed operational handoff.
