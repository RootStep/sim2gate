# v2 phase 0A: runbook

Phase 0A is the evidence Vihaan uses in phase 0B to decide whether ρ extends from BSRS to ordinary PPO reward
terms. It runs 10 Go2 training runs with the per-term recorder attached, measures the finished policies, and writes
one bundle. This page covers how to run it. The bundle's data format is in
[`src/sim2gate/v2/PHASE0A_FORMAT.md`](../../src/sim2gate/v2/PHASE0A_FORMAT.md), which also ships inside the bundle
as `FORMAT.md`.

Nothing in phase 0A is blinded or scored. The runs are labelled, because they feed a mathematical decision and do
not test Sim2Gate.

## What is new in the code

- `sim2gate.training.evidence.EvidenceRecorder` records, without changing training:
  - each reward term per step;
  - moments of [A, T_k, V-part];
  - gradient directions with each term removed;
  - the per-minibatch gradients and the actual update;
  - sample dumps, models, and critic refits.

  Tests: `tests/test_evidence.py`. These check bit-identical training, the identities, and an offline recomputation
  of the PPO gradient from the saved files.
- `isaaclab_launch` has `--evidence DIR` and related flags.
- `sim2gate.v2.phase0a` holds the arms, the two task variants (P0 v4 without `feet_slide`, and without
  `action_rate_l2`), and the driver. Tests: `tests/test_phase0a.py`.

## On the GPU instance (sim2gate-arena-2, inside `isaac-arena-vscode`, in `/workspaces/isaaclab_arena`)

1. Update Sim2Gate to the commit that has phase 0A, and reinstall it the same way as for the MVP run.
2. **Smoke test, about 10 minutes.** This is one 12-iteration run with every recorder feature on:

   ```
   python -m sim2gate.v2.phase0a smoke --work v2_runs/phase0a_smoke
   ```

   It should end with `SMOKE OK`. It also prints:
   - the reward term names;
   - the residuals: the term sum should be ~1e-7, the identity ≲ 1e-4, and the trainer ≲ 1e-6;
   - the refit losses;
   - the record size.

   Paste the output to Claude before starting the real runs.
3. **The 10 runs, about 5–6 hours.** The driver is resumable: rerunning it skips finished runs.

   ```
   python -m sim2gate.v2.phase0a train --work v2_runs/phase0a
   ```

   Each run is checked as it ends. The checks are that the record is complete, that the terms add up to the reward,
   that the identity holds, and that the recorded advantages are the trainer's. The run ends with `ALL RUNS OK`.
4. **Outcomes, about 1 hour.** These are check 2 (foot slip, jitter) and the 64-episode nominal walking check, on
   each final policy:

   ```
   python -m sim2gate.v2.phase0a outcomes --work v2_runs/phase0a
   ```

5. **Bundle:**

   ```
   python -m sim2gate.v2.phase0a bundle --work v2_runs/phase0a --out phase0a-evidence.tgz
   ```

   Note the printed SHA-256. Copy the bundle to VJ's computer and to the RootStep Drive folder, then send it to
   Vihaan.
6. Stop the instance.

GPU budget in the v2 plan: 8 hours for phases 0A–0B.

## Phase 0B

Vihaan analyses the bundle with his own code and decides one of three outcomes:

- ρ extends to ordinary reward terms as it stands;
- it needs a new derivation;
- it stays with shaping terms.

VJ approves the resulting plan change.
