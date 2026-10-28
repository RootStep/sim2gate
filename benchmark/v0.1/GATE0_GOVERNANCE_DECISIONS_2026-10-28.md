# Gate 0 governance decisions — 2026-10-28

**Authorized by RootStep representative:** proceed with technical Gate 0 closure only. **No benchmark freeze, held-out policy generation, training, GPU profiling, or confirmatory scoring is authorized.**

## Decision 1 — held-out label custody
The designated role is **Parent / RootStep company manager**. **Named custodian: Vijendra Kulkarni (parent / RootStep company manager), designated by user on 2026-10-28.** Private custody location, access boundary, and actual handoff are **not yet verified**. Vihaan Kulkarni remains technical lead and must not access the sealed held-out answer key during scorer development or tuning. This person must remain independent of scorer implementation, score tuning, and prediction generation.

- Custodian holds intervention assignments, run lineage, and any sealed answer key in a separate access-controlled location.
- The scorer receives only opaque random IDs and approved diagnostic measurements, not the key or its storage path.
- Before unsealing, commit predictions and their SHA-256, code revision, scoring configuration, and timestamp.
- After predictions are committed, custodian authorizes controlled unsealing for independent analysis. Preserve an access log and deviations.
- If the custodian participated in tuning, record the conflict and assign a different independent holder before confirmatory testing.

## Decision 2 — historical provenance exceptions
**Accepted:** recorded-only historical checkpoint hashes may remain in the exclusion inventory when original bytes cannot be recovered, **provided every such policy and all its related runs are categorically excluded**. No recorded-only hash may be described as byte-verified.

- Maintain source record, recorded digest if available, reason bytes are unavailable, and lineage/seed/recipe identifiers.
- If exact identity cannot be proven, exclude by broader run/recipe lineage rather than treating it as fresh.
- Full hashes and byte-verification should still be recovered opportunistically from already known original archives; this is not an indefinite hard blocker if conservative exclusion is demonstrated.
- Historical cohorts may support development analysis but not confirmatory held-out evidence.

## Remaining technical closure gates
1. Validate and audit the exclusion ledger and all lineage aliases; resolve incomplete SHA fields without guessing.
2. Execute automated input-boundary tests in a clean environment and archive passing logs and code revision.
3. Integrate the boundary into the actual v0.1 scorer and test synthetic metadata canaries, score invariance, logs and outputs, and key isolation end to end.
4. Record a named custodian, access boundary and dry-run custody evidence.
5. Obtain explicit independent approval of protocol taxonomy, calibration, baseline definitions and freeze artifacts.

**Decision: Gate 0 remains PARTIAL/BLOCKED pending actual test evidence and custody implementation.** These governance decisions remove two policy questions; they do not assert test success.
