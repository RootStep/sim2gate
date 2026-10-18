# v2 phase 0A: run record (Oct 18, 2026)

This pack is evidence for Vihaan's phase 0B decision: whether ρ extends from BSRS to ordinary PPO reward terms. It
is not a test of Sim2Gate and nothing in it is blinded or scored. The data format is in
[`src/sim2gate/v2/PHASE0A_FORMAT.md`](../../../../src/sim2gate/v2/PHASE0A_FORMAT.md); the runbook is in
[`docs/v2/PHASE0A.md`](../../PHASE0A.md).

## Run

- **Instance and software:** `sim2gate-arena-2` (L40S), Isaac Lab 91a0906, Isaac Lab-Arena 2cc9e97, rsl-rl-lib
  5.4.1.
- **Code:** sim2gate `cadc7b6`.
- **First attempt, earlier the same day (commit `d39a404`):** it crashed at the first PPO update in every run. The
  recorder built small helper tensors on the CPU next to CUDA data. The fix and a CPU-only test that catches this kind
  of bug are in `cadc7b6`. That attempt's folders were kept aside (`*.incomplete-*`) and are not in the bundle.
- **Smoke test** (12 iterations, every recorder feature): `SMOKE OK`. Record size 23 MB.
- **Training:** 10 runs of 2000 iterations, all `ok`. Each run passed the driver's checks: the record is complete,
  the reward terms add up to the reward, A = Σ_k T_k + V-part holds, and the recorded advantages equal the
  trainer's.
- **Outcomes:** check 2 (1000 steps, 64 environments) and the 64-episode nominal walking check on PhysX, for each
  final checkpoint. Every checkpoint hash matched.
- **Bundle:** `phase0a-evidence.tgz`, 691 files, 1004 MB, SHA-256
  `a893840106dfd70c42cd9b9aefdaa871f73605e9fb8fec0d6c822b678b929961`. It is kept outside the repo, on VJ's computer
  and in the RootStep Drive folder (Version 2).

## Outcomes (`outcomes.json`)

| Run | Foot slip p95 (m/s) | Jitter (HF power share) | Contact force p99 (body weights) | Nominal success (falls) |
|---|---|---|---|---|
| base_s1 | 0.104 | 0.00930 | 1.01 | 0.969 (1) |
| base_s2 | 0.101 | 0.00955 | 1.01 | 0.969 (0) |
| no_slip_s1 | 2.760 | 0.01353 | 0.69 | 1.000 (0) |
| no_slip_s2 | 2.810 | 0.01256 | 0.66 | 1.000 (0) |
| no_action_rate_s1 | 0.107 | 0.01224 | 1.02 | 0.922 (1) |
| no_action_rate_s2 | 0.100 | 0.01357 | 1.09 | 0.953 (0) |
| bsrs_0.5_s1 | 0.106 | 0.01063 | 1.05 | 0.969 (0) |
| bsrs_0.5_s2 | 0.103 | 0.00793 | 0.95 | 0.969 (0) |
| bsrs_2_s1 | 0.203 | 0.00973 | 1.06 | 0.953 (0) |
| bsrs_2_s2 | 0.161 | 0.00938 | 0.93 | 0.938 (2) |

Arm means against base (2 seeds each, so these are descriptive only):

| Arm | Foot slip | Jitter |
|---|---|---|
| no_slip | 27× base | +38% |
| no_action_rate | 1.0× | +37% |
| bsrs_0.5 | 1.0× | −1% |
| bsrs_2 | 1.8× | +1% |

## What the outcomes show

- **The anchors hold.** Without `feet_slide`, foot slip is about 27 times base (v1: about 25 times). Without
  `action_rate_l2`, jitter is about 37% above base. These jitter values (0.0122–0.0136) are the same as v1's P4.
  Every robot walks (92–100% nominal success).
- **The jitter is ambiguous.** Removing the slip penalty raised jitter as much as removing the action-rate penalty
  (+38% vs +37%). Jitter alone cannot tell which term was removed; foot slip can. This is the attribution problem the
  v2 Layer 2 study is about.
- **BSRS at η = 2 nearly doubled foot slip** (0.16–0.20 vs 0.10) without changing jitter. At η = 0.5 nothing
  changed. With 2 seeds this is an observation to look at in the per-term record, not a finding.
