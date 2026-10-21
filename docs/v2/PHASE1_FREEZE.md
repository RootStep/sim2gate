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
- **Oct 21, 2026, P0 v5 pilot at 914132f: PILOT FAIL.** Seed 1: nominal success 0.016 (0 falls; it stood still,
  speed-tracking reward 0.23 vs P0 v4's 1.36). Seed 2: 0.805 (0 falls). Neither qualified, so neither got check 1,
  check 2 or Newton. P0 v5 is not frozen. Why, and the lessons: `evidence/phase1/pilot_v5/README.md`.
- **Oct 21, 2026, revision P0 v5b (approved by VJ),** the first candidate in protocol section 3: the payload is drawn
  per episode, uniform 0 to an upper bound that is 0 kg before round 500 and 8 kg from round 800 (the delay's ramp);
  play mode uses the full 0–8 kg. Everything else, the pilot rule, the seeds and the success threshold are unchanged.
  Task `Sim2Gate-Velocity-Flat-Go2-P0-v5b`; robots `p0v5b_s{seed}`; new work folders. The smoke test also checks that
  the payload is recorded as per episode on that ramp.
