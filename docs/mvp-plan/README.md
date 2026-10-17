# Sim2Gate MVP plan: frozen copy

`MVP_PLAN_FROZEN_2026-10-12.md` is the Sim2Gate MVP design and validation plan exactly as Vihaan
approved it on Oct 12, 2026 (all 12 decisions ticked, no changes requested). It is the plan's
Markdown export, byte for byte, taken from the shared doc at revision 45 on Oct 13, 2026, while its
hash line was still blank.

Plan SHA-256: `9b9d2c858c3b791085480e50405960501cb34e9c158cffd800aa27f9dc805677`

Check it with `sha256sum -c SHA256SUMS` in this folder.

The plan's criteria are fixed from here on. Any later change is logged in this folder with its
date and reason, and never edits the frozen file.

Logged changes: [`CHANGES.md`](CHANGES.md).

Step 2 records: the current calibration in [`calibration/p0v4`](calibration/p0v4/README.md) (check 2 thresholds,
episode-success reference, checks 1 and 5 on P0 v4) and the planted-bug test of the validation in
[`evidence/planted-bugs`](evidence/planted-bugs/README.md).
The planted-flaw policies P1–P5 and the blinded-run procedure: [`test-set.md`](test-set.md).

**Result (Oct 17, 2026): the MVP did not meet three of its six preregistered acceptance criteria (A1, A2, A5 not
met; A3, A4, A6 met), so it is not validated.** Scores, verdicts, reports and the released key: [`evidence/step4`](evidence/step4/README.md).
