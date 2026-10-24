# v2 phase 2: freeze record

| Part | File | SHA-256 | Approved | Frozen before |
|---|---|---|---|---|
| A: arms, pilots, leak audit | `docs/v2/PHASE2_PART_A.md` | `d403db02227aca05d284db38ae0fa077f0a7e793204a3307551b36fb5101f8f7` | VJ, Oct 23, 2026 (revision 3 and the recorded details) | any pilot robot |
| B: study robots, inputs, ρ, rules, blinding, scoring | `docs/v2/PHASE2_PROTOCOL.md` | not yet frozen | — | any study robot |

Check: `sha256sum docs/v2/PHASE2_PART_A.md`. Later changes are listed below with date, reason and approval.

## Log
- Oct 24, 2026: arm pilots complete (16 robots, seeds 201–218 without 203–204).
  - R-slip passed at `feet_slide` × 0.1. Φ-slip, R-jitter, Φ-jitter and R-hop were dropped.
  - Under part A, C2 and C3 are not testable as designed. Details are in `evidence/phase2/pilots/README.md`.
  - Any revision is a new amendment, approved by VJ before any new robot is trained.
