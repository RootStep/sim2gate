# Step 4 and 5: the blinded run and its scoring (Oct 16–17, 2026)

**Result: the MVP validation failed its pre-registered acceptance criteria. A3, A4 and A6 passed; A1, A2 and A5
failed.** Nothing below was tuned after unblinding; the thresholds, success reference and rules are the frozen ones
(`../../FREEZE.md`, commit `4b25f92` with change C9).

## Run

- Instance `sim2gate-arena-2` (L40S), Isaac Lab 91a0906, Arena 2cc9e97, sim2gate 4b25f92 (`sim2gate-versions.txt`).
- Training (`step4_train.log`): 20 new policies, Oct 16 22:45 – Oct 17 12:29 ET. Retries under the fixed rules: P0 seed 5
  passed at seed 15 and P3-v2 seed 1 at seed 11 (seed + 10); P5's 2000-stiffness ground failed the walking check
  once, so all three P5 seeds moved to the 3000 fallback (`P5-v2`), as pre-declared, and one P5-v2 job passed on its
  third seed (seed + 20). The key records each policy's base seed; the training logs, sealed with Vihaan, show the
  seed used. P3 was
  redefined to P3-v2 (C9) after P3-v1 failed three times.
- Blinded run, seal and replays finished Oct 17 ~18:10 ET, exit code 0. Sealed digest
  `0edd2142cf6b2b2221ea827675a520c44b2496faf65ebc4e22c665cbbbe09fdb` (= SHA-256 of `verdicts.sha256`), recorded in
  `FREEZE.md` (commit 4f9e612) and sent to Vihaan before the key was released.
- Vihaan unsealed the key on his own computer (OpenSSL, his private key) and sent `key.json`. Its
  `manifest_sha256` and all 23 checkpoint hashes match `blind_manifest.json`.
- A known blinding gap (the run's console showed task reward terms) is recorded in `FREEZE.md`.

## Files

| File | What it is |
|---|---|
| `verdicts/`, `verdicts.sha256` | The 23 sealed verdicts and the sealed hash list (unchanged; its last line names `blind/manifest.json`, here `blind_manifest.json`) |
| `replay_verdicts/` | The three A6 reruns |
| `reports/` | Each policy's HTML report |
| `key.json` | The answer key, released after sealing |
| `acceptance.json` | `python -m sim2gate.mvp.acceptance --verdicts verdicts --key key.json --replays replay_verdicts` |
| `blind_manifest.json`, `blind_aliases.json`, `sealed_manifest.json` | The blind set and the sealed archive's hashes |

The raw measurements (61 MB) are in the bundle `s2g-step4-2026-10-17.tgz`, SHA-256
`4fd505cd85de78701b9ca96879ce1a911fd74fdeff01e555f621eb3249c9ab85`, kept outside the repo.

To check: `sha256sum verdicts.sha256` gives the digest above, and `cd verdicts/.. && sha256sum -c verdicts.sha256`
verifies each verdict (after copying `blind_manifest.json` to `blind/manifest.json`).

## Scores

| | Rule | Result |
|---|---|---|
| A1 | check 2 flags P3 and P4 in ≥ 2 of 3 seeds each | **Fail.** P3 3/3; P4 0/3 |
| A2 | check 1 flags friction or mass for P1, latency for P2, in ≥ 2 of 3 each | **Fail.** P2 3/3; P1 1/3 |
| A3 | check 5 flags P5 in ≥ 2 of 3 | **Pass.** 3/3 |
| A4 | ≤ 1 of 5 held-out P0 (seeds 4–8) gets no-go | **Pass.** 0/5 (and 0/8 counting seeds 1–3) |
| A5 | check 5 parity for every policy; every planted policy meets nominal success on PhysX | **Fail.** Parity held for all; P3 seed 2 (BU73H) nominal success 0.781 (check 1) / 0.828 (check 5) against 0.8 |
| A6 | 3 reruns reproduce their verdicts exactly | **Pass.** 3/3 |

## Per policy

| Policy | IDs (seed 1, 2, 3, …) | Verdicts | Flags |
|---|---|---|---|
| P0 | B681H, BG6L1, B6J1X, BU70R, BSGQ6, BEDC1, BB7B0, BGTBL | all go | none |
| P1 narrow randomization | B2SZQ, BDBSZ, BD7JV | all no-go | s1: friction, mass, motor strength; s2 and s3: motor strength only (cliff at 0.85, the threshold) |
| P2 no latency | BARCD, BY52S, BA6H6 | all no-go | latency cliff at 30 ms, all three |
| P3 sliding (P3-v2) | BLYV5, BU73H, BF88T | all no-go | check 2 foot slip p95 2.71–2.79 m/s (threshold 0.161), plus check 1 cliffs; s2 also check 5 |
| P4 jittery | B3CLP, BSCT3, BNFXJ | all go | none |
| P5 engine overfit (P5-v2) | B91TY, B6VA0, BDBHU | all no-go | check 5 error ratio 2.34–2.59 (threshold 2.0), plus check 2 slip/penetration and check 1 cliffs |

## Why the three criteria failed

- **A1, P4.** Check 2's jitter metric (high-frequency share of action power) was higher for every P4 seed
  (0.0122, 0.0136, 0.0128) than for every P0 seed (0.0095–0.0111), but below the frozen threshold 0.0162. Removing the
  action-rate penalty from the P0 v4 recipe made the policies only mildly jittery; the threshold, set from three
  calibration seeds, was too loose to call that a flaw. P2 (no latency training) also scored 0.0133–0.0157.
- **A2, P1.** All three P1 seeds were no-go, but two were flagged only on motor strength, which the pre-registered
  rule does not count. A narrower mass randomization showing up as sensitivity to weaker motors is physically
  plausible (both change how much load the legs carry), but that is an explanation after the fact, not a pass.
- **A5.** BU73H passed the training walking check (≥ 80% of 64 nominal episodes) and then scored 50/64 (78%) in check
  1's nominal condition. At 64 episodes a 95% interval around 80% is roughly ±10 points, so this is a borderline
  walker, not a broken check. Check 2 still caught its sliding.

## Descriptive only (not a criterion)

Counting any no-go as a detection: 12 of 15 planted policies flagged, 0 of 8 clean policies flagged. This framing was
chosen after unblinding and does not replace A1–A6.

## What is not allowed now

Changing the jitter threshold, the P1 rule or the walking bar and rescoring this run. Any change is a new plan change,
tested on newly trained policies.
