# Historical checkpoint follow-up backlog — register v2 (2026-10-30)

**Decision (unchanged from 2026-10-28):** RootStep has approved conservative exclusion of all historical policies, with recorded-only hashes and provenance exceptions documented. Recovery of missing historical checkpoint bytes and full hashes is **deferred**, not waived as a requirement for making any future positive identity claim. This backlog is for later reconciliation, **not a reason to reuse these policies as confirmatory held-out data**.

Source of record: [SEEN_POLICIES_PROVISIONAL.csv](SEEN_POLICIES_PROVISIONAL.csv), validated by [validate_seen_policies.py](validate_seen_policies.py). Do not infer missing hashes or treat recorded digests as byte-verified.

## What changed in register v2

- **97 rows** (was 60), all `SEEN_EXCLUDE`. New columns: `hash_source` (tracked file or operator record that supplies the digest), `aliases`, `notes`. The original 60 rows keep their order and values, except as listed below.
- **15 Phase 1 rows filled** (p0v5b_s107–s111, s113–s122): `UNRESOLVED` → `RECORDED_ONLY`. Digests are copied from `docs/v2/evidence/layer1/frozen_success_p0v5b.json` and `check2_limit_v2.json`, which agree for all 20 qualified seeds. These are recorded values, **not** byte-verified.
- **37 previously trained runs added** that the 60-row register omitted:
  - 20 v2 Phase 2 pilot robots (s201, s202, s205–s222): digests from the tracked pilot `state.json` files.
  - 3 MVP P0 v2 calibration robots: digests from `docs/mvp-plan/calibration/p0v2/manifest.json`.
  - 10 v2 Phase 0A runs and 4 Phase 1 P0 v5/v5b pilot robots: `UNRESOLVED`, because no digest is tracked.
- **Aliases recorded.** The P0 v4 seeds 1–3 checkpoints also appear as pre-freeze IDs `B9KTU`, `BV71C`, `BT7YY` and as p0v4 calibration seeds. P0 v2 seed 1 also appears as pre-freeze `BSUBF`, labelled `P2-standin`.
- **The validator now cross-checks every tracked digest source.** These are the MVP step 4 key, the Layer 1 files, both pilot state files, the p0v2 calibration manifest, plus the p0v4 and pre-freeze alias sources. It fails if any known digest is missing, misattributed, duplicated, or carries a guessed value.

## Remaining backlog

| Cohort | Items | Current evidence | Follow-up |
|---|---|---|---|
| v1 MVP | 23 | Full SHA-256 recorded only (step 4 key) | Re-hash `s2g-step4-2026-10-17.tgz` or original checkpoints; keep `RECORDED_ONLY` if unrecoverable |
| v2 Phase 1 | s107–s111, s113–s122 (15) | Recorded only (Layer 1 files) | Stream-hash final checkpoints from `phase1-clean.tgz` (on Drive per `GATE0_VERIFICATION_UPDATE.md`); upgrade to `BYTE_VERIFIED` only on match |
| v2 Phase 1 | s106, s112 (2) | `UNRESOLVED`; never measured | Same archive |
| v2 Phase 1 | s101–s105 (5) | Byte-verified (operator report) | None; reused as Phase 2 controls, count once |
| v2 Phase 2 | s301–s305, s309, s310 (7) | Recorded only; single source | Locate operator-generated `phase2_study_exclusion_ledger.csv` (not in any git ref; not found via Drive connector) and the Phase 2 raw export |
| v2 Phase 2 | s306–s308, s311, s312, s314, s315 (7) | Operator-reported byte match; **no digest or prefix stored** | Same ledger; never guess |
| v2 Phase 2 | s313 (1) | Incomplete run | Retain run-level exclusion; record its intermediate checkpoint lineage if recovered |
| v2 Phase 2 pilots | 20 | Recorded only (tracked `state.json`) | Re-hash from the pilot bundle; explain `train_returncode` 1 for s205, s210, s214; recover the first s216 attempt's digest |
| MVP P0 v2 calibration | 3 | Recorded only (tracked manifest) | Re-hash if the instance logs survive |
| v2 Phase 0A | 10 | `UNRESOLVED` | Extract digests from `phase0a-evidence.tgz` (recorded SHA-256 `a8938401…`) |
| v2 Phase 1 pilots (P0 v5, v5b) | 4 | `UNRESOLVED` | Extract from `phase1-pilot-v5.tgz`; identify the v5b pilot bundle |

**Counts:** 97 rows. Hash status: 68 `RECORDED_ONLY`, 5 `BYTE_VERIFIED`, 16 `UNRESOLVED`, 7 `BYTE_VERIFIED_SHA_PREFIX_ONLY`, 1 `INCOMPLETE_RUN`. 73 rows carry a full digest. 66 of those come from tracked repository files; the other 7 rest only on the untracked Phase 2 ledger.

**Before any future confirmatory run:** check every new checkpoint digest, seed, recipe and run family against this register. Runs without a known digest must be excluded by lineage.

**No training or protocol freeze authorized.**
