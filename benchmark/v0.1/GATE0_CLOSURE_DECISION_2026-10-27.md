# Gate 0 closure decision — 2026-10-27

**Decision: NOT PASSED.** Documentary reconciliation advanced; no protocol freeze, new confirmatory training, GPU profiling, or held-out scoring is authorized.

## Completed
- Google Drive root artifact register and v1/v2 folders located. The v2 Phase 1 clean archive (2,597,212,552 bytes) is present on Drive and locally reported SHA-256 matched to the recorded digest.
- Existing v1 ledger lists 23 historical seen policies with full recorded hashes, not independently rehashed bytes.
- A conservative provisional register `SEEN_POLICIES_PROVISIONAL.csv` now includes 23 v1, 22 v2 Phase 1, 14 completed Phase 2, and incomplete s313. Five Phase 2 reused controls are represented only under Phase 1. All rows are `SEEN_EXCLUDE`.
- Five Phase 1 reused controls have reported byte-verified checkpoint SHA matches.
- Seven Phase 2 final checkpoint files have reported matching nonempty bytes, but the provisional register includes only SHA prefixes for those seven; their full values must be copied from the previously generated `phase2_study_exclusion_ledger.csv`, not guessed.
- Phase 2 source archive incompleteness is documented; zero-byte placeholders must never be treated as valid checkpoints.
- Historic v1 and v2 thresholds are **not** a benchmark-wide calibrated 10% FAR.

## Outstanding blockers
1. Finish SHA-256 verification for the remaining 17 Phase 1 final checkpoints from the **original** `phase1-clean.tgz`. The manifest lists 616 checkpoint-related `.pt` files, but a previously inspected metadata extraction contained no checkpoint bytes. This does **not** imply the original archive lacks them.
2. Replace Phase 2 SHA-prefix placeholders with the seven complete byte-verified SHA values from the operator-generated Phase 2 exclusion ledger. Preserve seven hash-recorded-only entries and the incomplete s313 record.
3. Verify v1 raw archive bytes against recorded digest or preserve `RECORDED_ONLY` status with explicit limitation. Do not search endlessly for already known historical paths.
4. Perform a leakage test of **actual scorer inputs**: filename/path, training recipe, reward coefficients, policy IDs, intervention aliases, labels, and any diagnostic metadata. Include adversarial synthetic fixtures and ensure held-out labels are unavailable to the scorer.
5. Name an independent custodian and document sealed label access, release timing, prediction commitments, and handling of missing or retried runs.
6. Resolve protocol taxonomy inconsistency (five named categories described as six), operational definitions of each baseline, dense-grid reference, 10% calibration FAR, and power/compute plan. These are protocol-freeze requirements, not evidence that Gate 0 has passed.

## Leakage acceptance checklist (not yet executed)
- [ ] Inspect actual scoring input schema and serialization; permitlist only diagnostic features.
- [ ] Confirm source path, checkpoint filename, seed, recipe ID, intervention label, reward coefficients, and blinded key cannot be ingested.
- [ ] Plant distinct canary values in forbidden metadata; verify unchanged scores and no logging/export of canaries.
- [ ] Verify held-out key stored outside scorer runtime and inaccessible to model selection/tuning.
- [ ] Confirm all calibration thresholds fit **development-only** policies and are immutable before held-out scoring.
- [ ] Check policy-level split against full historical hashes and run lineage; handle unavailable hashes conservatively.
- [ ] Preserve test outputs, source commit, environment versions, and independent reviewer signoff.

## Explicit limits
The provisional register is **not** a claim of 59 unique completed policies. The 59 completed historical policy records are before cross-cohort deduplication; s313 is an additional incomplete attempt. Gate 1–6 technical checks are not scientific validation. The benchmark protocol remains DRAFT.
