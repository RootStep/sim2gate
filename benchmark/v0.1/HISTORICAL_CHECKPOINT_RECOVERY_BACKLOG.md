# Historical checkpoint follow-up backlog — deferred (2026-10-28; Phase 1 and Phase 2 updated 2026-10-30)

**Decision:** RootStep has approved conservative exclusion of all historical policies, with recorded-only hashes and provenance exceptions documented. Recovery of missing historical checkpoint bytes and full hashes is **deferred**, not waived as a requirement for making any future positive identity claim. This backlog is for later reconciliation, **not a reason to reuse these policies as confirmatory held-out data**.

Source of record: [SEEN_POLICIES_PROVISIONAL.csv](SEEN_POLICIES_PROVISIONAL.csv). Do not infer missing hashes or treat recorded digests as byte-verified.

| Cohort | Items | Current evidence | Follow-up |
|---|---|---|---|
| v1 MVP | 23 policies (see source register) | Full SHA-256 recorded only, not independently byte-verified | Reopen original archive, independently hash checkpoints and reconcile identities |
| v2 Phase 1 | p0v5b_s101–s105, s107–s111, s113–s122 (20 qualified) | **Byte-verified 2026-10-30** on the Brev instance; every digest matched at its Layer 1 run_dir | Done. See [phase1_checkpoint_byte_verification_2026-10-30.json](phase1_checkpoint_byte_verification_2026-10-30.json). s101–s105 were reused as Phase 2 controls; count them once |
| v2 Phase 1 | p0v5b_s106, p0v5b_s112 (2, not qualified) | No recorded digest | Identify their run folders under `logs/rsl_rl/sim2gate_go2_flat_p0_v5b/` on Brev by the seed in `params/agent.yaml`, then hash `model_1999.pt` |
| v2 Phase 2 | s301–s312, s314, s315 (14) | **Byte-verified 2026-10-30** against the decrypted sealed study archive; all 7 previously recorded digests matched, 7 missing digests recovered from bytes | Done. See [phase2_checkpoint_byte_verification_2026-10-30.json](phase2_checkpoint_byte_verification_2026-10-30.json). The external `phase2_study_exclusion_ledger.csv` is no longer needed |
| v2 Phase 2 | s313 (1) | Incomplete run | Retain run-level exclusion; never count as completed policy |

**Counts and interpretation:** 60 register rows, all marked SEEN_EXCLUDE. Of these, 35 have full SHA-256 populated (23 v1 recorded-only, 5 Phase 1 verified, 7 Phase 2 recorded-only); 25 do not (17 Phase 1 unresolved, 7 Phase 2 verified-prefix-only, 1 incomplete). This is a **provenance/evidence backlog**, not a claim that all 25 checkpoint files are physically missing. Some original archive bytes or ledger hashes may already be available. 59 rows correspond to completed runs before cross-cohort identity deduplication; s313 is incomplete.

**Priority:** Deferred until after the blinded scorer input boundary, synthetic leakage tests, and custodian dry run. All cohorts remain excluded. Before any future confirmatory run, audit recipe/run-family/seed aliases so unknown hashes cannot create an accidental historical overlap.

**No training or protocol freeze authorized.**
