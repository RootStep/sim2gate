# v2 phase 1: freeze record

| | |
|---|---|
| Frozen file | `docs/v2/PHASE1_PROTOCOL.md` |
| SHA-256 | `4a0ca34b87d7d8cdd3bb0cbabe75f5cf52fd2fefb6f6d0c4ad80598b712bb215` |
| Decided by | VJ, Oct 19, 2026 (P0 v5 trained across the full v2 envelope) |
| Phase 1 training before this commit | None |

Check: `sha256sum docs/v2/PHASE1_PROTOCOL.md`. Later changes (for example a P0 v5b after a failed pilot) are listed
below with date, reason and VJ's approval.

## Amendments

- **Oct 21, 2026, smoke test at 914132f.** Every measurement ran; the sweep was on the v2 envelope; the profile
  removed the payload; motor strength 0.8–1.0 was recorded. The one failed check was a reporting bug: Isaac Lab saves
  tuples as `!!python/tuple`, which the report's YAML reader turned into blanks, so the payload range read as None
  (v1 reports had the same blank for the stock mass range). Fixed for later runs (tuples read as lists, with a test).
  The payload range is confirmed directly from the smoke run's `params/env.yaml`. Neither the pilot rule nor training
  depends on this report field, so the pilot runs at 914132f. Approved by VJ.
