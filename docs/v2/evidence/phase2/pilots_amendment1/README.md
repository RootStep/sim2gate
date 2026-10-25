# Phase 2, amendment 1 pilots, Oct 25, 2026

**Result: both arms passed.** Under amendment 1 (`PHASE2_PART_A_AMENDMENT1.md`, SHA-256 `ea1ce36a…`) the cause study
is **testable**, with four arms: Control, R-slip (`feet_slide` × 0.1), Φ-slip-mild and Φ-delay. Before part B is frozen,
its symptom definitions are revisited as patterns across several measures (amendment 1, hard-stop section).

These 4 robots are development robots: excluded from all scoring, and never study robots or test robots.

Code `1784558`; the Isaac image was unchanged from earlier sessions (its ID was checked before and after setup).

## How the run went

- **First smoke attempt, Oct 24 evening: failed.** Both arms' training exited with code 1 at start-up, before Isaac
  Lab saved its settings. It ran right after a Sim2Gate Watch session on the same instance. The logs are kept on the
  instance (`v2_runs/phase2_smoke2/robots/*.incomplete-*`); the cause has not been read from them yet.
- **Second smoke attempt, Oct 25 morning: passed.** Both arms' saved configs matched their candidates, with the same
  11 reward terms. Nothing was changed between the two attempts.
- **Seed 219 was started twice.** The first start ran outside tmux and was stopped by hand about a minute in, so the
  run could not be lost to a dropped connection. It was retrained from the start; the partial folder is kept.
- The driver reports every robot. No robot had a config or run problem.

## Results

Qualify means at least 85% success over 256 episodes under the v2 threshold.

| Arm | Seed | Success | Falls | Qualifies | Intended symptom | Value |
|---|---|---|---|---|---|---|
| Φ-slip-mild (static friction 0.3–0.6) | 219 | 0.953 | 0 | yes | yes | slip p95 0.172 m/s (threshold 0.1185) |
| | 220 | 0.961 | 0 | yes | yes | slip p95 0.194 m/s |
| Φ-delay (no command delay) | 221 | 1.000 | 0 | yes | yes | check 1 flags latency; cliff 25 ms |
| | 222 | 1.000 | 0 | yes | yes | check 1 flags latency; cliff 25 ms |

**Pass rule (amendment 1):** both robots qualify and both show the intended symptom, with no problems. Both arms meet it.

## Measured values

From `state.json` (copied here; SHA-256 below). Check 2 in play mode; check 1 cliffs on the v2 envelope (`—` = no
cliff inside the range).

| Robot | Slip p95 (m/s) | Sustained slip p95 | Jitter share | Actuator share | Flight share, low speed | Vertical RMS, low speed (m/s) | Friction cliff | Latency cliff (ms) | Obs-noise cliff |
|---|---|---|---|---|---|---|---|---|---|
| Φ-slip-mild s219 | 0.172 | 0.101 | 0.00726 | 0.0042 | 0.0040 | 0.072 | 0.2 | — | 2.5 |
| Φ-slip-mild s220 | 0.194 | 0.102 | 0.00870 | 0.0036 | 0.0020 | 0.066 | 0.125 | — | 2.5 |
| Φ-delay s221 | 0.094 | 0.086 | 0.01064 | 0.0041 | 0.0148 | 0.075 | 0.275 | 25 | — |
| Φ-delay s222 | 0.095 | 0.085 | 0.01010 | 0.0031 | 0.0018 | 0.041 | 0.275 | 25 | — |
| For reference: R-slip k0.1 (s201, s202), first pilot round | 2.83, 2.69 | — | 0.0108, 0.0124 | 0.0025, 0.0029 | 0.0022, 0.0024 | 0.073, 0.065 | — | — | — |
| Controls 101–105 (maximum) | 0.106 | — | 0.00925 | 0.0027 | 0.0020 | 0.060 | — | — | — |
| Symptom threshold | > 0.1185 | — | > 0.00986 | — | ≥ 0.02 | ≥ 0.090 | — | in 0–30 ms | — |

## What the numbers add (reported only, found after the fact; no rule changes)

- **The two slip arms differ greatly in size.** R-slip slipped about 15× more than Φ-slip-mild (2.7–2.8 against
  0.17–0.19 m/s) and walked slightly better (99.6% against 95–96%). Behaviour measures alone (approach A) may
  therefore separate the reward-caused and physics-caused slip easily. C2 passes only if B or C beats A by 20
  points, so this works against C2. It bears on how part B's symptom patterns are defined.
- **Φ-delay robots were also above the jitter threshold** (0.0106 and 0.0101), as were R-slip's. Φ-slip-mild's were
  not (0.0073, 0.0087). Jitter again moved with changes other than the action-rate penalty, as in the first round.
- **Φ-slip-mild robots kept walking at lower friction** in check 1 (cliffs 0.2 and 0.125, against 0.275 for Φ-delay),
  as expected from training on lower friction. Neither was flagged.
- **No hopping:** every robot was below the hop thresholds. Φ-delay seed 221 had the highest flight share at low
  speed (0.0148), still under the 0.02 floor.

`state.json` SHA-256: `93453a06d824c2f1cba446e9b18c8a65826ecaa798f75759dd058114212fac97`. The full bundle (training
records and checkpoints) is kept outside the repository: `phase2-pilot2.tgz`, 304 files, 472 MB, SHA-256
`c882dcd1b76f071f3b47495e534acc4d6bc39fad6005ea1cbe76f709420b6bab` (made Oct 25 on `sim2gate-arena-2`).

## What happens next

No further pilot rounds (amendment 1). Before part B is frozen:

1. Part B is revised for the four arms, with symptom definitions as patterns across several measures.
2. The leak audit (part A) is run on the pilot robots' records.
3. Vihaan confirms C1 and signs off phase 0B Q1 and Q4.

VJ approves the revised part B before any study robot is trained.
