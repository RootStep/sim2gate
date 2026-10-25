# Phase 2 leak audit (part A), Oct 25, 2026

**Result: passed.** Run on `sim2gate-arena-2` with code `24784f2` (`python -m sim2gate.v2.phase2_export audit`), on
the training records of all 20 pilot robots (both pilot rounds; 10 arm settings × 2 robots). Pilot robots only: no
study robot or control was read.

| Condition (PHASE2_PART_A.md) | Result |
|---|---|
| 1. Same reward-term names, order and format in every pilot record | **Pass.** One term list (11 terms) in all 20, and every update's per-term array has 11 entries |
| 2. No primary input (B, C or B-var feature) constant within every arm while differing between arms | **Pass.** None found, so no feature is removed |
| 3. Trivial clue: nearest centroid on weight × step_dt per term, leave one robot out (reported only) | **Accuracy 0.60** (chance 0.10) |

**Reading condition 3.** The 10 reward-arm robots (R-slip, R-jitter × 2 settings, R-hop × 2 settings) are each
identified exactly by their weights. The 10 physics-arm robots and the base recipe share one set of weights, so the
weights cannot tell them apart. 12 of 20 correct is what that predicts. This is why weights and weighted
contributions stay out of the primary inputs: with them, every reward arm would be given away.

**ρ under section 6's rules.** 440,000 update–term values (20 robots × 2000 updates × 11 terms): **0 undefined,
0 invalid.** Every update passed the identity check A = Σ_k T_k + V.

File: `phase2_leak_audit.json`, SHA-256 `1c821fcf8d3307ac54f34fafa950a8ee9fd0f7ca6171020c2799e0a3693e9c62`.
