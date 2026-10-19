# v2 phase 0B: decision protocol (DRAFT, not frozen)

**Status:** draft by VJ and Claude, Oct 18, 2026, for Vihaan to change and fill in. It takes effect when Vihaan
has filled the blanks and VJ has approved. Its SHA-256 is then recorded here before any of the analyses below are
run on the phase 0A data.

## 1. The decision

Does ρ, derived in paper V1.4 for same-critic BSRS, extend to the ordinary reward terms of a PPO task? The plan
allows three answers:

- **(a) It extends as it stands.**
- **(b) It needs a new derivation.**
- **(c) It stays with shaping terms.**

Under (c), Layer 2 continues with approach B (standard training diagnostics) only, and the report says so.

**Who decides:** Vihaan decides, from his derivation and the phase 0A numbers, using his own code. VJ approves the
resulting plan change. Claude prepared the data and this draft, and does not make the call.

**What phase 0A already settled, and what it did not:** phase 0A recorded the data and showed that removing
`feet_slide` and `action_rate_l2` changes the finished robots as expected (`evidence/phase0a/README.md`). It did not
test whether ρ predicts anything for ordinary terms. Nobody has computed the update-level quantities below on the
real runs yet. Keeping it that way until this protocol is frozen means the criteria are set before the answer is
known.

## 2. Scope: what phase 0B can and cannot decide

**It can decide** whether ρ, or a per-term extension of it, describes how each reward term shapes **each PPO
update** on the Go2 task. That is the level V1.4 works at, and phase 0A recorded it for every iteration of 10 runs.

**It cannot decide** whether a per-term ρ predicts **which behaviour a term causes in the finished robot**, or
attributes a symptom to the right term. With 2 seeds per arm and two removed terms there is nothing to score that
on. That is the phase 2 Layer 2 study (C2–C3), with 5 seeds per arm and held-out scoring. Phase 0B decides only
whether phase 2 uses ρ (approach C) or not.

## 3. Questions, in order

### Q0. Reference: does ρ do on Go2 what V1.4 says it does for BSRS?

V1.4's identity was verified on a captured Go2 rollout. Its link from ρ to the gradient, Adam, KL and multi-iteration
training was not yet tested there. Phase 0A covers it: ρ at η = 0.5 and 2 in the two BSRS arms, and the what-if ρ at
η = 0.5, 1 and 2 in every run, next to the corresponding update directions. **If ρ does not predict the update change
for BSRS itself, the extension question does not arise.**

### Q1. Algebra: does V1.4's closed form carry over to removing a term?

This is an observation for Vihaan to confirm or correct in his derivation, not a result. Sim2Gate's `tail_predictor`
(the same algebra as his package) writes the shaped advantage as A' = (1+η)A − ηT. Regressing T on A
(T = βA + R) gives

ρ = c‖A‖ / √(c²‖A‖² + η²‖R‖²), with c = 1 + η − ηβ.

The same steps apply to any A' = aA − bT_k. Removing term k is a = b = 1, so with T_k = β_k A + R_k:

ρ_k = corr(A, A − T_k) = (1 − β_k)‖A‖ / √((1 − β_k)²‖A‖² + ‖R_k‖²).

Claude checked this numerically on synthetic data, not on the phase 0A runs: it matches corr(A, A − T_k) to 6 decimals, including the negative values when β_k > 1. It gives the same regimes: removing a term flips the update when β_k > 1, and cancels as an affine change when
R_k = 0. If this holds, the per-rollout statistic itself needs no new derivation. **What the algebra does not cover is
listed in Q3 and Q4.**

### Q2. Update level: does ρ_k track the change in the update as well as ρ does for BSRS?

Compare ρ_k (`corr_A_A_minus_term`, or ρ_k from `moments`) with:

- the cosine between the trainer's direction and the direction without term k: `direction_cos_trainer["minus:<term>"]`;
- how the actual update aligns with each: `update_cos`;
- optionally, per-minibatch gradients from the directions files.

Do this at the deep iterations of all 10 runs, for every term, and judge the fit **against the Q0 reference**. The
question is whether ordinary terms behave like shaping terms at this level.

### Q3. The critic: is the same-critic counterfactual a fair stand-in?

BSRS keeps the critic by construction. Removing a term for real retrains the critic on a different reward. `base_s1`
has critic refits without `feet_slide` and without `action_rate_l2`, at iterations 50, 200, 500, 1000 and 1999:
`refit.<term>.corr_A_minus_term` (agreement between A − T_k and the refitted advantages), and the
`refit_minus:<term>` update directions. If the refit moves the update far from the same-critic version, ρ_k
describes the wrong counterfactual. That points to (b).

### Q4. Not about the update: what ρ_k cannot mean, whatever Q0–Q3 show

BSRS shaping is potential-based, so in theory it leaves the optimal policy unchanged and ρ measures only
optimization distortion. An ordinary term changes the objective itself. ρ_k can at most describe how much the term
steers each update. It says nothing by itself about where training ends up. Vihaan to state in the memo what ρ_k may
and may not be claimed to mean.

## 4. Decision rules (Vihaan fills the blanks before any analysis)

| Answer | Condition |
|---|---|
| **(a) Extends as it stands** | Q0 holds (criterion: ______), Q1 holds, Q2's ρ_k fit is within ______ of the Q0 reference, and Q3's refit agreement is at least ______ at ______ of the refit points |
| **(b) Needs a new derivation** | Q0 and Q2 hold but Q3 fails, or Q2 holds only with a different functional form (described: ______) |
| **(c) Stays with shaping terms** | Q2 fails: ρ_k does not track the update change within ______ |
| **Q0 fails** | ρ's own Go2 link is not confirmed; report it, with V1.4 implications, and treat as (c) for v2 |

Also fixed in advance:

- which statistic measures "fit" (for example rank correlation, or residual of the closed form);
- which iterations count (all deep iterations, or after a warm-up);
- how the two seeds are combined;
- what counts as inconclusive, and what happens then.

## 5. Where each quantity is in the bundle

| Quantity | Location (see FORMAT.md) |
|---|---|
| ρ for BSRS (actual, BSRS arms; what-if, all runs) | `updates.jsonl` → `bsrs_what_if_rho`; also `monitor.jsonl` |
| ρ_k, β_k, R_k | `updates.jsonl` → `moments` (covariance of [A, T_1…T_K, V-part]); `corr_A_A_minus_term` |
| Change in update direction | deep iterations: `direction_cos_trainer`, `direction_norm`, `update_cos`; vectors in `deep/*_directions.npz` |
| KL, clip fraction, learning rate | `updates.jsonl` |
| Critic refits | `base_s1` at the refit iterations: `refit`, `refit_minus:<term>` |
| Raw samples for any recomputation | `deep/*_samples.npz`, `deep/*_models.pt` |
| Behaviour outcomes (descriptive only) | `outcomes/`, and the run record |

A caution on wording: FORMAT.md inside the bundle calls corr(A, A − T_k) a "ρ-like" quantity. It is an analogue,
recorded for convenience, and its meaning is what Q1–Q4 decide.

## 6. Output of phase 0B

1. **Vihaan's decision memo:** (a), (b) or (c), with his numbers for Q0–Q3 and the Q4 statement, against the frozen
   rules.
2. **Optional independent cross-check:** Sim2Gate computes the same Q0–Q3 tables with its own code, only after the
   protocol is frozen, and the two are compared. Agreement between two implementations is stronger evidence than
   either alone. Vihaan decides whether he wants this.
3. **VJ approves the plan change.** The v2 plan's Layer 2 then names approach C (with ρ) or B only, and C1 records
   Vihaan's confirmation of the Oct 9 RSL-RL gate.
