# Gate 0 — evidence and threshold inventory (Oct 27, 2026)

**Status:** documentary inventory from tracked GitHub records. External bundles and their hashes have **not** been independently downloaded or verified in this audit. No new training or scoring performed.

## Existing cohorts

| Cohort | Recorded size | Usage and caveat | Primary record |
|---|---:|---|---|
| MVP v1 P0 v4 + P1–P5 | 23 total: P0 seeds 1–8, P1–P5 three each | **Seen**; 3 P0 calibration, 5 P0 originally held out in v1; now all development-only | `docs/mvp-plan/evidence/step4/README.md` |
| v2 Phase 1 P0 v5b | 22 trained, **20 qualified** | Existing calibration reference, not a new confirmatory set; 106 and 112 did not qualify | `docs/v2/evidence/phase1/clean/README.md` |
| v2 Phase 2 | 20 assigned, **19 completed measurement** | Intervention study; 313 failed twice and has no behaviour measures | `docs/v2/PHASE2_CONCLUSION.md` |

### v1 blinded evidence
- Frozen code recorded as commit `4b25f92`; sealed verdict digest `0edd2142cf6b2b2221ea827675a520c44b2496faf65ebc4e22c665cbbbe09fdb`.
- `docs/mvp-plan/evidence/step4/` contains tracked key, verdicts, replay verdicts, reports and manifests as described by its README.
- External raw bundle: `s2g-step4-2026-10-17.tgz`, recorded SHA-256 `4fd505cd85de78701b9ca96879ce1a911fd74fdeff01e555f621eb3249c9ab85`, recorded raw measurement size 61 MB. **Bundle itself not verified here.**
- Acceptance **failed A1, A2, A5**; passed A3, A4, A6. Descriptive any-no-go detection 12/15 planted and 0/8 clean is **post hoc**, not the preregistered endpoint.
- Known blinding gap: console exposed task reward terms; must not replicate.

### v2 Phase 1 clean evidence
- Qualified seeds: `101–105,107–111,113–122`; excluded 106 and 112 for not meeting 0.85 nominal success.
- External `phase1-clean.tgz`: recorded SHA-256 `36bb9b75caf2ab7d37482bf7dc79c5ddd48cacfd2bde3f3aac834ef5c23b712a`, 2,597 MB, 1,704 files.
- External `phase1-clean-measurements.tgz`: recorded SHA-256 `9e1be1705da1073e1bd5a68cc2fe5469846ead13cae5663390b4d081be5fb38e`, 1.4 MB, includes manifest and 82 measurements.
- External `phase1-clean-curves.tgz`: recorded SHA-256 `3e50c16b1c950226f33cb23aaeb534a019b97da6d5c7f60880206389726cba65`, 52.9 MB.
- `src/sim2gate/v2/layer1_limits.py` records calibration seed list and verifies measurement hashes against manifest; requires external extracted files.
- `docs/v2/evidence/layer1/README.md` reports 1/20 clean flagged by check 1 under recorded v1 success threshold, 0/20 by leave-one-out check 2, 0/20 check 5; do not treat these as independently verified new cohort FARs.

### v2 Phase 2 evidence
- `docs/v2/evidence/phase2/scoring/` tracks feature table, key, predictions, score and study report.
- Original 20-row analysis is **not protocol-compliant** because seed 313 had no behaviour measurements. Protocol-compliant recount: behaviour 17/19; training diagnostics 19/19; training plus rho 19/19; **C2 and C3 not met**.
- Recount retains out-of-fold predictions originally trained with seed 313 in other folds, so it is not a clean refit. Preserve this limitation.
- External phase-2 study bundle availability and integrity **not confirmed** by this repository-only inventory.

## Threshold provenance — **do not mix**
| Version | Calibration population | Threshold / rule | Applicability concern |
|---|---|---|---|
| v1 P0 v4 | Three P0 calibration seeds | v1 episode tracking error threshold **0.1133 m/s**; check 2 per-metric thresholds under P0 v4 measurement settings | 3-seed calibration, different recipe and measurement configuration |
| v2 P0 v5b | 20 qualified clean runs | Episode tracking error **0.1187 m/s** (no fall); combined check 2 worst-metric ratio **>1.0541** | Different recipe, check 2 actuator clipping fix, different metric aggregation |
| v2 check 1 | v2 envelope and success definition | Flag cliff within realistic condition range; nominal success and thresholds depend on evaluation settings | Not a calibrated 10% FAR detection score |
| v2 check 5 | PhysX/Newton | Newton fall-free share <80% or tracking error >2× PhysX | Engine configuration and fall definition must match |

**V2 combined check 2 calibration maxima:** slip 0.1124 m/s; penetration 0.00352 m; actuator-limit step share 0.00366; jitter high-frequency share 0.00935. Contact force report-only. Values are copied from tracked evidence narrative; this audit did not recompute them.

**Critical conclusion:** Neither v1 nor v2 thresholds automatically implement the proposed benchmark-wide **10% calibration FAR**. That endpoint needs a new calibration protocol and scored outputs. Avoid applying thresholds across measurement versions without explicit validation.

## Reproducibility and leakage audit checklist
- [x] Locate tracked v1 key/verdicts and documented hashes
- [x] Locate tracked v2 clean-cohort protocol, evidence narrative and threshold computation code
- [x] Locate tracked phase-2 scoring narrative and documented protocol deviation
- [ ] Independently verify external v1 raw bundle bytes/hash
- [ ] Independently verify v2 Phase 1 clean and measurement bundle bytes/hash
- [ ] Locate/verify v2 Phase 2 complete external evidence export
- [ ] Inventory all checkpoint SHA-256 values and actual storage locations
- [ ] Execute read-only tests / reproducibility commands in pinned environment
- [ ] Test metadata leakage, including task aliases, reward terms and paths
- [ ] Establish policy-level split ledger for all previously seen seeds/checkpoints
- [ ] Specify independent held-out custodian and permitted feature access

## Gate decision
**Gate 0 partially complete, not passed.** Documentary inventory and threshold mismatch are established; external artifact verification and full split ledger remain open. No held-out GPU training or benchmark freeze authorized.

## Sources
- `docs/mvp-plan/evidence/step4/README.md`
- `docs/v2/evidence/phase1/clean/README.md`
- `docs/v2/evidence/layer1/README.md`
- `docs/v2/evidence/phase2/scoring/README.md`
- `docs/v2/PHASE2_CONCLUSION.md`
- `docs/v2/LAYER1_LIMITS_PROTOCOL.md`
- `src/sim2gate/v2/layer1_limits.py`
