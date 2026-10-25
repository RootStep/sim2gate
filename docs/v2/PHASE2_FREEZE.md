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
- Oct 24, 2026: **amendment 1 approved by VJ and frozen**, before any of its robots was trained.
  - File: `docs/v2/PHASE2_PART_A_AMENDMENT1.md`, SHA-256 `ea1ce36a6d6bbaf9dbd6bb3825a4032e94159e799f5043668e6befcd6ec07e71`.
  - Content: the final bounded pilot round, Φ-slip-mild (seeds 219–220) and Φ-delay judged by check 1's latency
    sweep (seeds 221–222).
  - Hard stop: if Φ-slip-mild fails, C2 and C3 are not tested.
- Oct 25, 2026: amendment 1 pilots complete (4 robots, seeds 219–222), code `1784558`.
  - Φ-slip-mild and Φ-delay both passed. The cause study is testable with Control, R-slip, Φ-slip-mild and Φ-delay.
  - Details in `evidence/phase2/pilots_amendment1/README.md`.
  - Part B is next revised for these arms (symptom patterns, leak audit), then frozen after Vihaan's sign-offs and
    VJ's approval, before any study robot.
- Oct 25, 2026: **leak audit passed** on the 20 pilot robots (code `24784f2`); see `evidence/phase2/leak_audit/`.
  Part B revision 4 (review corrections; key holder confirmed) remains unfrozen until the scoring program is tested
  and VJ approves the final text.
