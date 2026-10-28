# Historical checkpoint follow-up backlog — deferred (2026-10-28)

**Decision:** RootStep has approved conservative exclusion of all historical policies, with recorded-only hashes and provenance exceptions documented. Recovery of missing historical checkpoint bytes and full hashes is **deferred**, not waived as a requirement for making any future positive identity claim. This backlog is for later reconciliation, **not a reason to reuse these policies as confirmatory held-out data**.

Source of record: [SEEN_POLICIES_PROVISIONAL.csv](SEEN_POLICIES_PROVISIONAL.csv). Do not infer missing hashes or treat recorded digests as byte-verified.

| Cohort | Items | Current evidence | Follow-up |
|---|---|---|---|
| v1 MVP | 23 policies (see source register) | Full SHA-256 recorded only, not independently byte-verified | Reopen original archive, independently hash checkpoints and reconcile identities |
| v2 Phase 1 | p0v5b_s106 through p0v5b_s122 (17) | Full SHA-256 missing / unresolved | Recover final checkpoint bytes from phase1-clean archive; hash each member and verify lineage |
| v2 Phase 2 | s301, s302, s303, s304, s305, s309, s310 (7) | Full SHA-256 recorded only | Verify original bytes against recorded digests |
| v2 Phase 2 | s306, s307, s308, s311, s312, s314, s315 (7) | Byte-verified previously but register contains only SHA prefix status, not full digest | Recover full SHA-256 from existing phase2_study_exclusion_ledger.csv and reconcile; do not guess |
| v2 Phase 2 | s313 (1) | Incomplete run | Retain run-level exclusion; never count as completed policy |
| v2 Phase 1 | p0v5b_s101 through p0v5b_s105 (5) | Full SHA-256 byte-verified | No immediate recovery required; these controls were reused in Phase 2 and must not be counted twice |

**Counts and interpretation:** 60 register rows, all marked SEEN_EXCLUDE. Of these, 35 have full SHA-256 populated (23 v1 recorded-only, 5 Phase 1 verified, 7 Phase 2 recorded-only); 25 do not (17 Phase 1 unresolved, 7 Phase 2 verified-prefix-only, 1 incomplete). This is a **provenance/evidence backlog**, not a claim that all 25 checkpoint files are physically missing. Some original archive bytes or ledger hashes may already be available. 59 rows correspond to completed runs before cross-cohort identity deduplication; s313 is incomplete.

**Priority:** Deferred until after the blinded scorer input boundary, synthetic leakage tests, and custodian dry run. All cohorts remain excluded. Before any future confirmatory run, audit recipe/run-family/seed aliases so unknown hashes cannot create an accidental historical overlap.

**No training or protocol freeze authorized.**
