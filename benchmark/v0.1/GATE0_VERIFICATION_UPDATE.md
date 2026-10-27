# Gate 0 — verification update (Oct 27, 2026)

**Status: PARTIAL — Gate 1 blocked pending byte-level checks and complete policy ledger.** This update adds direct connected Google Drive discovery and GitHub key inspection to the earlier narrative inventory.

## Directly located on Google Drive (metadata only, not cryptographic validation)

| Artifact | Drive file ID | Size in Drive | Recorded expected SHA-256 | Status |
|---|---|---:|---|---|
| v1 `s2g-step4-2026-10-17.tgz` | `1OGkqMcabXwHbV_uINZxUq78jSqW4RndL` | 278,967,410 bytes | `4fd505cd85de78701b9ca96879ce1a911fd74fdeff01e555f621eb3249c9ab85` | Exists; hash not verified |
| v2 `phase1-clean.tgz` | `1_OhPDQV0ZTx26NN8F_QATTwc7tsf1pCJ` | 2,597,212,552 bytes | `36bb9b75caf2ab7d37482bf7dc79c5ddd48cacfd2bde3f3aac834ef5c23b712a` | Exists; hash not verified |
| v2 `phase1-clean-measurements.tgz` | `1CgtW8Valiv-SV_HHN4sIT69b9zn2hVHA` | 1,379,676 bytes | `9e1be1705da1073e1bd5a68cc2fe5469846ead13cae5663390b4d081be5fb38e` | Exists; hash not verified |
| v2 `phase1-clean-curves.tgz` | `1mIsCQYqUCQGGnL_7Wr8PCrUmzK4WcNc7` | 52,864,828 bytes | `3e50c16b1c950226f33cb23aaeb534a019b97da6d5c7f60880206389726cba65` | Exists; hash not verified |

Drive v1 folder: https://drive.google.com/drive/folders/17We6Wpj_3itI_49qoi9vFKto_Mh2tg4P

Drive v2 clean folder: https://drive.google.com/drive/folders/1QkJ73HXsR5Dh5kL5z9lQlKjOFWdrkP4x

**Phase 2:** Drive folder `phase2_blind` exists at https://drive.google.com/drive/folders/1Zedf-xnDFdoNarq-nnAlRQXUj8suVxk8 with `features.json` (67,611 bytes), `report.json` (447 bytes), and `sealed/sealed_for_vihaan.key.enc` (384 bytes). This is **not evidence of a complete raw study export**; raw checkpoints, training logs and measurements were not located there.

## Checkpoint ledger
- The tracked v1 key has **23 entries, 23 distinct IDs and 23 distinct recorded checkpoint hashes**. See [SEEN_MVP_V1_LEDGER.md](SEEN_MVP_V1_LEDGER.md).
- P0=8, P1–P5=3 each. All must be excluded from confirmatory benchmark.
- The original checkpoint paths point to an instance workspace; **existence of the checkpoint bytes at those paths is not verified**.
- The v2 Phase 1 clean 20 qualified seeds are listed in `GATE0_EVIDENCE_INVENTORY.md`, but a checkpoint-level SHA ledger still requires the external `phase1-clean.tgz` state/manifest.
- Phase 2 19 measured and seed 313 failed; checkpoint-level SHA ledger requires the raw study export.

## CI verification
- Run `40533630161` (commit `60c9de9`) **passed**.
- Run `40533452630` failed due to an overly broad test assertion matching digits `313` in a random SHA-256 digest, not an actual exposed seed.
- Commit `91b260d` narrows the assertion; run `40534319053` was **in progress when inspected**. Do not claim it passed until its completion is verified.

## Outstanding requirements for a Gate 0 pass
1. Compute SHA-256 **over actual downloaded bytes** for all four listed archives and compare with expected digests. Drive metadata and past reports are insufficient.
2. Read `MANIFEST.json` inside clean measurements; verify 82 measurements, 20 checkpoint mappings, and per-file digests against available raw records.
3. Locate complete Phase 2 raw export and verify manifest/bytes, or document irrevocable absence and downgrade reproduction claims.
4. Extract v2 clean and phase-2 policy/checkpoint hashes, training seeds, recipe versions and any retry replacements into a complete **seen policy ledger**.
5. Execute clean-environment reproducibility checks; review leakage paths in method inputs, not just random aliases.
6. Record which required evidence is on Drive versus instance/PC and identify the custodian.

**No Gate 1 approval, held-out policy generation or threshold reuse until these are resolved.**

## Local WSL byte-verification addendum — Oct 27, 2026

**Evidence source:** user-executed read-only SHA-256 and tar inspection commands in WSL; outputs reviewed in conversation. This is a reported local verification, not an independent rerun by the repository CI.

- `phase1-clean.tgz`: computed SHA-256 `36bb9b75caf2ab7d37482bf7dc79c5ddd48cacfd2bde3f3aac834ef5c23b712a` — matches previously recorded expected digest.
- `phase1-clean-measurements.tgz`: computed SHA-256 `9e1be1705da1073e1bd5a68cc2fe5469846ead13cae5663390b4d081be5fb38e` — matches recorded digest.
- `phase1-clean-curves.tgz`: computed SHA-256 `3e50c16b1c950226f33cb23aaeb534a019b97da6d5c7f60880206389726cba65` — matches recorded digest.
- `phase1-pilot-v5.tgz`: computed SHA-256 `acda5d8cd4bc6560f64eba317c9cc5fd804b6e6dd195233574a8a83ac0693f2e`; no independently recorded expected digest compared.
- The main Phase 1 archive contains `phase1_clean/robots/p0v5b_s101` through `s122` final checkpoint entries (22 entries; qualification is separate).
- Direct streaming SHA-256 of the five original Phase 1 `checkpoint/model_1999.pt` members matched the checkpoint fingerprints recorded consistently in the Phase 2 control `nominal.json`, `sweep.json`, and `newton.json` records:

| Reused control | Verified checkpoint SHA-256 | Result |
|---|---|---|
| p0v5b_s101 | `14e03fab96f671e5b45c43ffefe947ea11754c6a07634d232c58b67745befe62` | MATCH |
| p0v5b_s102 | `dcd188cbf70d0c9c0a94cd1dd2165e714c977c9db67382563900f96358e0989b` | MATCH |
| p0v5b_s103 | `612ea5b6f092e44b338fa74d88b2d9c8befb3481e0202c9026295edc7d941343` | MATCH |
| p0v5b_s104 | `7c3061069ae3571e8b11c1719db4a578f24844e156745c66968a2c34d7ea2215` | MATCH |
| p0v5b_s105 | `4891931732af39e55fee00cd9ea7aa5c99b664969837b3fccadd4d19f579e46e` | MATCH |

**Phase 2 raw inventory (reported local inspection):** 21 folders; 426 checkpoint-file entries, 402 distinct checkpoint digests. The 24 duplicate entries come from the two seed-313 incomplete/retry folders, which share their 24 checkpoint hashes; seed 313 is not a completed measured study robot. Fourteen completed study robots have final checkpoints and measurement records. Five controls have evaluation records but reference original Phase 1 checkpoints rather than embedding them.

**Scope:** This closes the specific five-control checkpoint-byte identity check. It does not independently authenticate when earlier expected archive hashes were recorded, validate evaluation results, establish checkpoint runtime reproducibility, or qualify any historical policy as fresh held-out benchmark data. Previous 'hash not verified' statements above describe the earlier inspection state and are superseded **only for the three locally hashed Phase 1 archives**. Gate 0 remains PARTIAL; no protocol freeze or Gate 1 approval.
