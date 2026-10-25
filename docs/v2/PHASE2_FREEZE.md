# v2 phase 2: freeze record

| Part | File | SHA-256 | Approved | Frozen before |
|---|---|---|---|---|
| A: arms, pilots, leak audit | `docs/v2/PHASE2_PART_A.md` | `d403db02227aca05d284db38ae0fa077f0a7e793204a3307551b36fb5101f8f7` | VJ, Oct 23, 2026 (revision 3 and the recorded details) | any pilot robot |
| B: study robots, inputs, ρ, rules, blinding, scoring (revision 4) | `docs/v2/PHASE2_PROTOCOL.md` | `2d1b0a632bdde78796d481fa784371569532897bb02c1af92e78f12898db28ba` | VJ, Oct 25, 2026 (in place of Vihaan's pending sign-offs, through the documented review; the three later text changes approved) | any study robot |
| B: scoring program | `src/sim2gate/v2/phase2_score.py` | `7231e36cc005aa15870f2bda484f4c40fdf681ff8ce42d5dcfd0f8fb8b126d92` | with part B | any study robot |
| B: feature exporter | `src/sim2gate/v2/phase2_export.py` | `24a9490103af26653ec7a407ce97cac46ef5ede9722a0da86e62193dcbc755a9` | with part B | any study robot |

Check: `sha256sum docs/v2/PHASE2_PART_A.md docs/v2/PHASE2_PROTOCOL.md src/sim2gate/v2/phase2_score.py src/sim2gate/v2/phase2_export.py`. Later changes are listed below with date, reason and approval.

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
- Oct 25, 2026: **part B frozen (revision 4), approved by VJ.** Before any study robot is trained.
  - Preconditions met: pilots (amendment 1 passed); leak audit passed; gradient cross-check passed (756 directions,
    minimum cosine 0.9999999999999996); key holder and scoring operator confirmed (Vihaan; v1 key `9db064c8…`).
  - VJ's Oct 25 authorisation stands in for Vihaan's pending personal sign-offs (C1, phase 0B Q1 and Q4, and the
    ★ decisions), through a documented review whose corrections are in part B's "Review corrections".
  - The three text changes after that approval (scoring program named, four implementation details, bootstrap
    strata wording) were shown to VJ and approved before the freeze.
  - Frozen with part B: the scoring program (`7231e36c…`) and the feature exporter (`24a94901…`).
  - Still open, not blocking: the original Oct 9 gate report (being searched for); Vihaan's OK for any public naming
    of his package.
  - Next: the study driver (training, control re-measure, random IDs, export, sealing), tested, then a GPU session.
