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

## 4. Decision rules

### 4.1 Proposed rules (VJ and Vihaan, Oct 18, 2026)

| Test | Proposed rule |
|---|---|
| Q0: BSRS reference | Confirm that ρ predicts gradient and update changes in the BSRS runs, using the original analysis method of V1.4 |
| Q1: Mathematical identity | Relative numerical error ≤ 10⁻⁵ where applicable; independently reproduced gradient cosine ≥ 0.999999 (see 4.3: on the 256-environment sample) |
| Q2: Ordinary reward terms | Compare per-term ρ with measured gradient and update changes, using Spearman correlation, against simpler diagnostic baselines named in advance (4.2) |
| Q3: Critic refitting | Compare same-critic predictions with directions from refitted critics; report agreement at every available refit point |
| Q4: Interpretation | Explicitly separate effects on PPO updates from effects on final robot behaviour |

### 4.2 Two thresholds, each a separate decision fixed before any analysis

Neither is to be chosen by looking at the phase 0A numbers. Both are written as rules, so their cut-offs come
from quantities defined in advance, not from the answer.

**T1. How much weaker per-term ρ may be than the BSRS reference (Q2 against Q0).** Options:

- *T1-a, non-inferiority margin (recommended).* Per-term ρ passes if its Spearman correlation with the measured
  update change is at most δ below the BSRS reference, with the 95% interval of the difference above −δ. δ = ____
  (suggested 0.1). Resample **runs, not iterations** (iterations within a run are correlated); with 10 runs the
  interval is wide, and a wide interval is reported as inconclusive, not as a pass.
- *T1-b, absolute floor plus baseline.* Spearman ≥ ____ (suggested 0.7) **and** ρ beats every simple baseline in 4.2
  by a margin whose 95% interval excludes zero.
- Simple baselines, named now: the term's share of reward magnitude (`term_abs_share`), Var(T_k)/Var(A) from
  `moments`, and β_k alone. If ρ does not beat these, it adds nothing a standard diagnostic doesn't.

**T2. How much critic-refit disagreement is acceptable (Q3).** Options:

- *T2-a, PPO's own noise as the yardstick (recommended).* The refit is acceptable at a refit point if
  cos(g_refit_minus_k, g_minus_k) is at least the median cosine between PPO's per-minibatch gradients and the
  full-rollout trainer direction at that iteration (`minibatch_grad_cos_trainer`). In words: removing the term with
  a refitted critic changes the direction less than PPO's own minibatch sampling does. The cut-off is fixed by rule
  now and read from a different quantity, so it is not tuned to the answer.
- *T2-b, fixed floor.* cos ≥ ____ at ≥ ____ of the 10 refit points (2 terms × 5 iterations).
- Either way all 10 points are reported, and with one run (`base_s1`) Q3 is a check, not a statistical test.

| Answer | Condition |
|---|---|
| **(a) Extends as it stands** | Q0, Q1 pass; Q2 passes T1; Q3 passes T2 |
| **(b) Needs a new derivation** | Q0, Q1, Q2 pass but Q3 fails T2; or Q2 holds only with a different functional form |
| **(c) Stays with shaping terms** | Q2 fails T1 |
| **Q0 fails** | ρ's own Go2 link is not confirmed: report it, with V1.4 implications, and treat as (c) for v2 |
| **Inconclusive** | Intervals too wide to pass or fail: report as inconclusive; v2 proceeds with approach B, and C stays an open question |

Also fixed in advance: which iterations count (all deep iterations, or after a warm-up of ____), and how the two
seeds are combined (pooled, with runs as the resampling unit).

### 4.3 Two gaps in the data the rules must respect

- **The gradient check in Q1 has to use the sample.** The saved directions are over the whole rollout (4096
  environments); the sample dumps hold 256. A full-rollout direction cannot be reproduced from the sample, so
  "independently reproduced gradient cosine ≥ 0.999999" means: two implementations (Vihaan's and Sim2Gate's)
  computing g on the same 256-environment sample from the same `models.pt` agree to that cosine. Comparing a
  sample direction with the full-rollout one measures sampling, not correctness.
- **There is no "null" refit.** Phase 0A refitted the critic without each term, but not with nothing removed, so the
  refit's own drift is not measured. T2-a avoids needing it. If a null refit is wanted, it can be run offline on the
  256-environment samples and saved models at the five refit iterations (null, minus `feet_slide`, minus
  `action_rate_l2`, like for like), with no GPU. That must be decided now, before Q3 is looked at.

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
