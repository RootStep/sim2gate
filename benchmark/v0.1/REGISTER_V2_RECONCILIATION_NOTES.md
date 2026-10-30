# Historical exclusion register v2 — reconciliation notes

This 97-row register is a **historical exclusion inventory**, not a benchmark
cohort, qualification claim, or evidence of checkpoint-byte verification.

* 60 original rows retained with SEEN_EXCLUDE.
* 15 Phase 1 hashes copied from tracked Layer 1 records as RECORDED_ONLY.
* 20 Phase 2 pilot hashes copied from two tracked pilot state files.
* 3 P0 v2 calibration hashes copied from the tracked calibration manifest.
* 10 Phase 0A and 4 Phase 1 pilot rows are **provisional run-level placeholders**,
  not independently recovered checkpoint identities. Their IDs are locally
  assigned inventory labels, **not original checkpoint filenames**. The
  original run-to-seed mapping and checkpoint hashes still require archive
  reconciliation. Exclude those cohorts by recipe/lineage until then.
* The seven Phase 2 study hashes still await the operator-held
  phase2_study_exclusion_ledger.csv. No digest was inferred.
* All 97 rows remain SEEN_EXCLUDE. Existing BYTE_VERIFIED statuses were
  preserved from historical operator records; this change does not re-hash bytes.

The validator compares registered digests against tracked source records.
It cannot establish independent archival provenance or completeness of
untracked historical runs. Alias and retraining lineage review remain open.

Custody changes on this branch are synthetic engineering tests only.
No real held-out labels, keys, freeze, or confirmatory evaluations were used.
