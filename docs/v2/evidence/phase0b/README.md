# v2 phase 0B: results and comparison of the two analyses (Oct 18, 2026)

**Status: provisional.** The answer under the frozen rules is below. Vihaan still has to sign off three things: Q1's
derivation, the check of his code against Sim2Gate's on the 256-environment sample, and the Q4 statement. Until then
nothing here is a claim about ρ.

## 1. Answer under the frozen rules

`sim2gate.v2.phase0b` (commit `12dfca1`) ran on the fingerprinted bundle (a8938401…) under
`PHASE0B_PROTOCOL.md` and amendment 2. Its output is in `phase0b_report.md` and `phase0b_results.json`.

**Answer: (a) extends as it stands.** The same answer holds without iterations 0–49.

| Test | Result | Rule |
|---|---|---|
| Q0, BSRS reference | Spearman 0.911 (95% CI 0.864–0.948), 486 pairs | ≥ 0.7, lower bound > 0.5: pass |
| Q1, numerics | max relative error 2.6e-6 (A), 2.1e-7 (terms), 2.4e-15 (closed form) | ≤ 1e-5: pass |
| Q2/T1, per-term ρ against reference | 0.953 vs 0.911; difference +0.042 (CI +0.014 to +0.081) | lower bound > −0.1: pass |
| Baselines | ρ beats reward share (+0.16), β_k (+0.17) and Var(T_k)/Var(A) (**+0.002**, CI +0.001 to +0.002) | lower bound > 0: pass |
| Q3/T2, critic refits | 0 of 10 points below the yardstick | ≤ 2 below: pass |

## 2. The two analyses agree on the numbers

VJ shared a separately run exploratory analysis (`independent/`, received after `12dfca1` was committed). Its
pooled Spearman values (0.482 for terms, 0.469 for BSRS) cover **all 10 runs**. The frozen analysis uses only the
**6 PPO runs**. In the 4 BSRS-trained runs the trainer's own direction is the shaped one, so "cosine with the
trainer" there does not measure removing a term from the PPO update.

Restricted to the same 6 runs, the independent CSVs give **exactly** the frozen numbers: 0.953 (1,674 pairs) and
0.911 (486 pairs). Two separate implementations reading the bundle agree. The difference was only which runs were
included.

## 3. What the frozen rules did not test (post hoc, exploratory)

Everything in this section was computed **after** the frozen answer was known, from the independent CSVs. It is
reported so that the frozen answer is not over-read, and it does not change that answer.

**a. The pooled pass mostly reflects term size.** A single pooled Spearman across all terms is driven by between-term
ordering: big terms such as velocity tracking move the update a lot, and small terms barely move it. Var(T_k)/Var(A)
alone ranks them almost as well (−0.951 against ρ's 0.953). ρ beats it by 0.002. That passes the rule, but in
practice ρ adds almost nothing to that ranking.

**b. Tracking within one term over training is mostly good, but fails for `feet_slide`.** Spearman within each term,
over the deep iterations of the 6 PPO runs:

| Term | Pooled | Median within run |
|---|---|---|
| track_lin_vel_xy_exp | 0.951 | 0.914 |
| track_ang_vel_z_exp | 0.948 | 0.916 |
| ang_vel_xy_l2 | 0.937 | 0.946 |
| action_rate_l2 | 0.941 | 0.924 |
| dof_acc_l2 | 0.920 | 0.906 |
| flat_orientation_l2 | 0.839 | 0.805 |
| lin_vel_z_l2 | 0.827 | 0.796 |
| dof_torques_l2 | 0.823 | 0.696 |
| feet_air_time | 0.502 | 0.333 |
| **feet_slide** | **0.120** | **0.053** |
| dof_pos_limits | constant (zero) | — |

For `feet_slide`, the term whose removal changed behaviour most (27 times more slip), ρ does not track the change
in the update. It also underestimates the change: the median 1 − ρ is 0.012, but the median 1 − cos is 0.032. The
range is not compressed (1 − ρ runs from 0.006 to 0.096), so this is not a ceiling effect.

ρ_k is a correlation in advantage space, and it ignores which state–action features the term acts through. A term
that is small overall but concentrated on specific features (foot contacts) can turn the update more than its
correlation suggests. This is an observation for Vihaan's derivation, not a conclusion.

**c. The T2 yardstick turned out to be uninformative.** PPO's per-minibatch gradients have a median cosine of about 0
(between −0.02 and 0.03) with the full-rollout direction, so T2 could only fail for a cosine near zero. Resolved Oct 20: this is a real property of PPO's updates, not a recording error. The recorded first-minibatch gradient equals the clipped-surrogate gradient computed directly (cosine 1.000000; now a test in tests/test_evidence.py). On a toy task with 512 environments, the first minibatch steps align with the full-rollout direction (cosines 0.68–0.97). After a few Adam steps the policy has moved, and the later steps scatter around zero or turn negative, so a median over all 20 steps sits near zero. The yardstick should have used only the first step of each update, which measures sampling noise alone. That is a design flaw in T2-a, which Claude proposed. It is reported here, not rescored. The offline null refit is more informative. Refitting the critic **without
removing anything** moves the direction about as much as refitting without a term:

| Iteration | null | − feet_slide | − action_rate_l2 |
|---|---|---|---|
| 50 | 0.905 | 0.890 | 0.911 |
| 200 | 0.872 | 0.850 | 0.869 |
| 500 | 0.955 | 0.957 | 0.956 |
| 1000 | 0.892 | 0.888 | 0.893 |
| 1999 | 0.922 | 0.916 | 0.912 |

So most of the "critic issue" here is the refit's own drift, not the removal. This is on 256 environments, offline,
for one run.

**d. "Update change" means the gradient direction, not a replayed optimizer step.** Amendment 2 fixed the measure
as the cosine of same-critic gradient directions. The independent report points out that no counterfactual
optimizer update (Adam state, clipping, several epochs) was replayed. That is a limit of phase 0A's data, not
something phase 0B can fix.

## 4. What this means for v2

- Under the frozen rules the answer is (a). Section 3 shows that the rules (a pooled Spearman, which Claude
  proposed) were too weak to test what matters most: whether ρ follows **each** term over training. For the
  term that mattered most here, it did not.
- The v1 principle applies: the frozen result is not rescored and the rules are not changed after the fact. Any
  stronger claim needs a new, preregistered test on new runs. For example: within-term tracking with runs as the
  unit, and every planted term required to pass.
- The decision on what to claim and what to carry into phase 2 is Vihaan's (Q1 derivation, Q4), approved by VJ.
