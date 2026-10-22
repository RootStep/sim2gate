# v2 Layer 1 limits: freeze record

| | |
|---|---|
| Frozen file | `docs/v2/LAYER1_LIMITS_PROTOCOL.md` |
| SHA-256 | `3b72a48590c337160832c1862926e5bdd7b9984d38f5ff4d90ca964671498a18` |
| Approved by | VJ, Oct 22, 2026 (all five decisions as written) |
| Clean robots' check 2 or Newton values read before this commit | None. Read so far: nominal success and check 1 flags from the run log, and seed 101's torque-clipped share (to confirm the check 2 fix) |

Check: `sha256sum docs/v2/LAYER1_LIMITS_PROTOCOL.md`. Later changes are listed below with date, reason and VJ's
approval; none may change a limit after the analysis has run.

## Log

- Oct 22, 2026: also approved, keep criterion C3 (ρ against standard diagnostics) in phase 2. It is computed from the
  same training records, with no extra robots or GPU time.
- Oct 22, 2026: computed once (commit `5d1696a`) from `phase1-clean-measurements.tgz` (`9e1be170…`).
  - Results: success threshold 0.1187 m/s, combined check 2 limit 1.0541.
  - Clean robots with a no-go: 1 of 20; the stop rule was not triggered.
  - Details: `evidence/layer1/README.md`.
