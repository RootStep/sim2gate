# Phase 0A evidence pack: data format (format_version 1)

For Vihaan. The pack holds 10 Go2 flat-ground training runs on Isaac Lab + RSL-RL 5.4.1, recorded by
`sim2gate.training.evidence.EvidenceRecorder`. Use it to decide (phase 0B) whether ρ, derived for same-critic BSRS
in paper V1.4, extends to the ordinary reward terms of a task. Everything is plain JSON, NumPy `.npz` and PyTorch
`.pt`; you do not need Sim2Gate's code to read it. Your ppo-shaping-diagnostics package was not used and is not
included.

The recorder only observes. With and without it, final parameters are bit-identical (`tests/test_evidence.py`, CPU).
Training is RSL-RL's own PPO, or Sim2Gate's `BSRSPPO` for the two BSRS arms. `BSRSPPO` matches the reference
implementation that passed the acceptance gate against rsl-rl-lib 5.0.1, 5.4.1 and 5.5.1, down to the actor
gradient.

## Runs

| Run | Task | Training |
|---|---|---|
| `base_s1`, `base_s2` | P0 v4 (Go2 flat + foot-slip penalty, friction and delay randomization, 2000 iterations) | PPO (= BSRS η = 0) |
| `no_slip_s1`, `no_slip_s2` | P0 v4 without `feet_slide` | PPO |
| `no_action_rate_s1`, `no_action_rate_s2` | P0 v4 without `action_rate_l2` (as v1's P4) | PPO |
| `bsrs_0.5_s1`, `bsrs_0.5_s2` | P0 v4 | same-critic BSRS, η = 0.5 |
| `bsrs_2_s1`, `bsrs_2_s2` | P0 v4 | same-critic BSRS, η = 2 |

Answers known from v1, for anchoring: removing `feet_slide` raised foot slip about 25-fold, and removing
`action_rate_l2` raised action jitter about 25%. Each run's `outcomes/` holds the finished policy's measured values.

## Notation

Per iteration, the rollout has T = 24 steps × N = 4096 environments. Actor and critic are MLPs [128, 128, 128] (ELU); the actor outputs a Gaussian over 12 joint actions. γ = 0.99 and λ = 0.95 (exact values are in
`meta.json`).

- r_t is the reward the environment returned, before RSL-RL's time-out bonus. r_t = Σ_k r_t^k, where r_t^k is
  reward term k's contribution, `term value × weight × step_dt`, as Isaac Lab's RewardManager adds it.
- V is the trainer's critic, evaluated when the step was collected (RSL-RL's stored values). On a time-out step,
  RSL-RL's GAE adds a bootstrap of γV(s_t). On a termination, it bootstraps 0.
- **A** is the GAE of r with V and RSL-RL's time-out convention. This is the trainer's unnormalized advantage for
  PPO arms.
- **T_k** is the discounted tail of term k: GAE with V = 0, T_t^k = Σ_j (γλ)^j r_{t+j}^k within the episode
  segment.
- **V-part** is the GAE of zero reward, the critic's own contribution.
- GAE is linear in the rewards for fixed values, so **A = Σ_k T_k + V-part** exactly. **A − T_k** is the advantage
  with term k removed, using the same critic.
- **T** = Σ_k T_k is the tail of the whole reward. Same-critic BSRS trains on **A' = (1+η)A − ηT**, and ρ = corr(A, A').
- **Â** is an advantage after RSL-RL's normalization, (x − mean)/(std + 1e-8) over the whole rollout
  (`normalize_advantage_per_mini_batch` is false for this task).

## Files per run (`runs/<run>/`)

| File | Content |
|---|---|
| `train.log` | Isaac Lab's training output |
| `monitor.jsonl` | Sim2Gate's TrainingMonitor, one line per iteration: rollout ρ and the what-if signals at η = 1 (as in experiment T) |
| `evidence/meta.json` | Run settings: γ, λ, clip, epochs, minibatches, num_envs, steps, **`reward_terms` (order of every per-term array)**, step_dt, network shapes, the iterations below, versions |
| `evidence/updates.jsonl` | One JSON line per iteration, below |
| `evidence/deep/itNNNNN_samples.npz` | Sample dump at each deep iteration, below |
| `evidence/deep/itNNNNN_models.pt` | `{"actor", "critic"}` RSL-RL state_dicts **before** that iteration's update |
| `evidence/deep/itNNNNN_directions.npz` | Gradient vectors at the grad-save iterations, below |
| `outcomes/check2.json`, `outcomes/nominal.json` | Layer 1 measurements of the final policy (check 2: foot slip, jitter; nominal: 64 episodes) |
| `checkpoint/model_1999.pt` | The final policy |

The top-level `MANIFEST.json` gives each file's SHA-256, together with the driver's state: per run, its return
code, checkpoint hash and outcome summary.

**Deep iterations:** 0, 1, 2, 5, 10, 20, 50, 100, 200, …, 1900, 1999.
**Grad-save iterations:** 0, 50, 200, 500, 1000, 1999.
**Critic refits:** `base_s1` only, at iterations 50, 200, 500, 1000 and 1999, for `feet_slide` and `action_rate_l2`.

## `updates.jsonl` (every iteration)

| Key | Meaning |
|---|---|
| `iteration`, `deep` | Iteration index (0-based), and whether it is a deep iteration |
| `term_step_mean`, `term_step_abs_mean` | Per term: the mean and mean-absolute r^k over the rollout |
| `term_abs_share` | `term_step_abs_mean` / its sum |
| `term_sum_residual` | max \|Σ_k r^k − r\| over the rollout (should be ~1e-7) |
| `identity_residual` | max \|A − (Σ_k T_k + V-part)\| (float32 recursion; should be ≲ 1e-4) |
| `trainer_residual` | max \|trainer's A (or A' under BSRS) − ours\| / max(1, max \|trainer's\|) |
| `moments.mean`, `moments.cov` | Mean vector and covariance matrix (float64, ddof 1) of the columns `[A, T_1 … T_K, V-part]` over all T·N samples. Column order is `meta.moments_columns`. Any correlation, variance ratio or regression among A, T_k, A − T_k, A − Σ_{k∈S} T_k or (1+η)A − ηT follows from these |
| `corr_A_A_minus_term` | Per term: corr(A, A − T_k), the "ρ-like" quantity for removing term k |
| `bsrs_what_if_rho` | `{"0.5", "1", "2"}`: corr(A, (1+η)A − ηT). At the training η of a BSRS arm, this is that run's actual ρ |
| `timeouts`, `terminations` | Counts in the rollout |
| `kl_subset`, `clip_fraction_subset` | Mean KL(old ‖ new) and the share of \|ratio − 1\| > clip, after the update, on the first 256 environments' samples |
| `learning_rate`, `losses` | After the update (adaptive schedule); RSL-RL's loss dict |

Deep iterations add these keys:

| Key | Meaning |
|---|---|
| `direction_norm`, `direction_cos_trainer` | For each variant below, the norm of g(Â) = ∇_θ mean(Â · log π_θ(a\|s)) over the whole rollout at the pre-update actor, and its cosine with the trainer's g. PPO's clipped-surrogate gradient at ratio 1 is −g(Â_trainer) (checked in tests). g is an ascent direction |
| `update_norm`, `update_cos` | The actor's actual parameter change Δθ over the whole update (all epochs, minibatches, Adam, gradient clipping), and its cosine with each variant's g |
| `minibatch_grad_norm`, `minibatch_grad_cos_trainer` | Per optimizer step (5 epochs × 4 minibatches = 20): the norm of the actor's loss gradient after gradient clipping (what Adam receives; it includes the entropy bonus, coefficient 0.01), and the cosine of its negative with g(trainer) |
| `refit` (refit iterations) | Per refitted term: the value loss per round, corr(A, A_refit) and corr(A − T_k, A_refit) |

Variants (names in `direction_*`, `update_cos` and the directions file):

| Name | Â |
|---|---|
| `trainer` | The advantages the trainer actually used (A under PPO, A' under BSRS), normalized |
| `A` | Normalized A |
| `minus:<term>` | Normalized A − T_k, the same critic with term k removed |
| `bsrs:<η>` | Normalized (1+η)A − ηT for η = 0.5, 1, 2 |
| `refit_minus:<term>` (refit iterations) | Normalized GAE of r − r^k with a critic refitted to r − r^k |

The refit is fitted value iteration on a copy of the trainer's critic: 3 rounds, each recomputing GAE targets on
r − r^k with the current copy, then 5 epochs of Adam (lr 1e-3) on squared error, with RSL-RL's minibatch count. It
is warm-started, so it shows how far the critic must move when a term is removed, not a critic trained from scratch.

## `deep/itNNNNN_samples.npz`

These cover the first 256 environments for all 24 steps, as [T, 256, …] float32 arrays, before the update.

| Array | Shape | Meaning |
|---|---|---|
| `term_names` | [K] | Reward term order |
| `obs_policy` (one `obs_<group>` per observation group; this task has only `policy`, used by actor and critic) | [T, S, 48] | Observations as the networks receive them (this task turns observation normalization off) |
| `actions` | [T, S, 12] | Actions taken |
| `old_log_prob` | [T, S] | log π(a\|s) at collection (equal to the pre-update actor's, since observation normalization is off) |
| `old_dist_param_0`, `old_dist_param_1` | [T, S, 12] | Gaussian mean and std at collection |
| `values`, `last_values` | [T, S], [S] | V(s_t), and V of the observation after the last step |
| `raw_rewards`, `terms` | [T, S], [T, S, K] | r and r^k |
| `stored_rewards` | [T, S] | r plus RSL-RL's γV time-out bonus (what RSL-RL stores) |
| `dones`, `time_outs` | [T, S] | 1 at episode ends, and at time-limit ends |
| `trainer_advantages_unnormalized`, `trainer_advantages` | [T, S] | The trainer's advantages before and after its normalization (normalization statistics come from the full rollout) |

Recomputing A from these arrays: GAE backwards over t. δ_t = r_t + γ·next_t − V_t, where next_t = V_{t+1}
(`last_values` at t = T−1) normally, V_t itself on a time-out step, and 0 on a termination. Then
A_t = δ_t + γλ(1 − done_t)A_{t+1}. `tests/test_evidence.py` does exactly this and checks it against the trainer.

## `deep/itNNNNN_directions.npz` (grad-save iterations)

| Array | Shape | Meaning |
|---|---|---|
| `names` | [M] | Variant names, as above |
| `directions` | [M, P] | g(Â) for each variant, over the whole rollout (P = actor parameters, in `actor.parameters()` order) |
| `actual_update` | [P] | Δθ of this iteration's update |
| `minibatch_loss_grads` | [20, P] | The actor's loss gradient at each optimizer step, after clipping |
| `param_shapes` | [n] | Shapes for unflattening |

## What is not included, and why

- **Per-minibatch normalization groups:** not used by this task.
- **Gradients of the critic:** the question concerns the actor's update direction, and the critic's state is in
  the models file.
- **Every sample at every iteration:** about 1 GB per iteration. The moments in `updates.jsonl` carry the
  second-order statistics of every iteration, and the deep dumps carry the samples.
