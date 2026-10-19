# Phase 0B results (provisional until Vihaan signs off Q1 and Q4)

Bundle SHA-256 a893840106dfd70c42cd9b9aefdaa871f73605e9fb8fec0d6c822b678b929961. Rules: docs/v2/PHASE0B_PROTOCOL.md (frozen) and amendment 2.

## Answer under the frozen rules: **(a) extends as it stands**

- All deep iterations: (a) extends as it stands
- Without iterations 0-49: (a) extends as it stands

## Q0, BSRS reference (6 PPO runs)

Spearman(rho, cos) = 0.911, 95% CI 0.864 to 0.948, 486 pairs. Rule: >= 0.7 and lower bound > 0.5: **pass**.

BSRS arms, descriptive (training-eta rho vs cos with g(A)):

- bsrs_0.5_s1: eta 0.5, mean rho 0.915, mean cos 0.927, Spearman 0.918 (n=27)
- bsrs_0.5_s2: eta 0.5, mean rho 0.913, mean cos 0.922, Spearman 0.892 (n=27)
- bsrs_2_s1: eta 2, mean rho 0.694, mean cos 0.708, Spearman 0.731 (n=27)
- bsrs_2_s2: eta 2, mean rho 0.690, mean cos 0.666, Spearman 0.565 (n=27)

## Q1, numerics on the sample dumps

270 dumps. Max relative error: A_vs_trainer 2.6e-06, terms_vs_reward 2.1e-07, closed_form_rho_k 2.4e-15. Rule <= 1e-05: **pass**. The derivation and the cross-implementation gradient check are Vihaan's.

## Q2 and T1, per-term rho against the reference

- All deep iterations: Spearman per-term 0.953 vs reference 0.911; difference 0.042, 95% CI 0.014 to 0.081 (margin -0.1): **pass** (1674 pairs)
- Without 0-49: Spearman per-term 0.984 vs reference 0.961; difference 0.022, 95% CI 0.012 to 0.043 (margin -0.1): **pass** (1302 pairs)

Baselines (|S_rho| - |S_baseline|, all deep iterations):

- term_abs_share: Spearman -0.791, difference 0.161, 95% CI 0.134 to 0.192: pass
- var_T_k_over_var_A: Spearman -0.951, difference 0.002, 95% CI 0.001 to 0.002: pass
- beta_k: Spearman -0.780, difference 0.173, 95% CI 0.161 to 0.184: pass

## Q3 and T2, critic refits (base_s1)

| Iteration | Term | cos(refit, same critic) | Yardstick (median minibatch cos) | Below |
|---|---|---|---|---|
| 50 | feet_slide | 0.964 | -0.024 | no |
| 50 | action_rate_l2 | 0.970 | -0.024 | no |
| 200 | feet_slide | 0.936 | 0.009 | no |
| 200 | action_rate_l2 | 0.942 | 0.009 | no |
| 500 | feet_slide | 0.940 | -0.007 | no |
| 500 | action_rate_l2 | 0.939 | -0.007 | no |
| 1000 | feet_slide | 0.891 | 0.025 | no |
| 1000 | action_rate_l2 | 0.879 | 0.025 | no |
| 1999 | feet_slide | 0.863 | 0.034 | no |
| 1999 | action_rate_l2 | 0.853 | 0.034 | no |

0 of 10 below the yardstick (fails if more than 2): **pass**.

Offline null refit (context only; cos of refitted vs same-critic direction on the 256-env sample):

- it 50: null 0.905, minus:feet_slide 0.890, minus:action_rate_l2 0.911
- it 200: null 0.872, minus:feet_slide 0.850, minus:action_rate_l2 0.869
- it 500: null 0.955, minus:feet_slide 0.957, minus:action_rate_l2 0.956
- it 1000: null 0.892, minus:feet_slide 0.888, minus:action_rate_l2 0.893
- it 1999: null 0.922, minus:feet_slide 0.916, minus:action_rate_l2 0.912

## Not decided here

- Q4: what per-term rho may and may not be claimed to mean (Vihaan).
- Whether per-term rho predicts behaviour in the finished robot: phase 2, not phase 0B.
