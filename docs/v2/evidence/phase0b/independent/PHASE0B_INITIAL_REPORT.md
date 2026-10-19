# RootStep Sim2Gate Phase 0B — initial exploratory analysis

Date: 2026-10-18/19. Status: partial analysis, not a final decision or preregistered result.

## Data integrity
Original split archive verified SHA-256 `a893840106dfd70c42cd9b9aefdaa871f73605e9fb8fec0d6c822b678b929961`. Parsed 10 run histories, 2,000 iterations each, 27 deep iterations each. Max reported identity residual across runs < 2.0e-6; trainer residual < 2.5e-7.

## Exploratory method
For each recorded deep iteration, compared `1-corr_A_A_minus_term` with `1-direction_cos_trainer['minus:<term>']`, excluding undefined/missing terms. For BSRS reference, compared `1-bsrs_what_if_rho[eta]` with `1-direction_cos_trainer['bsrs:<eta>']`. Computed pooled and term-specific Spearman coefficients. These are same-critic counterfactual *gradient directions*, not actual alternative PPO optimizer updates or behavior. Repeated within-run observations are dependent. Results below are descriptive and not inferential.

## Full-history exploratory results
Ordinary terms: 2,862 usable term/iteration observations, pooled Spearman 0.482. BSRS: 810 observations, pooled Spearman 0.469. Pooled across different terms/etas is confounded by term identity, eta and run arm; this is not a valid standalone noninferiority test.

Per-term Spearman (full history): `track_lin_vel_xy_exp` .684; `track_ang_vel_z_exp` .444; `lin_vel_z_l2` .261; `ang_vel_xy_l2` .526; `dof_torques_l2` .288; `dof_acc_l2` .731; `action_rate_l2` .202 (216 observations); `feet_air_time` .027; `flat_orientation_l2` .445; `dof_pos_limits` undefined (constant); `feet_slide` .137 (216 observations). BSRS reference eta .5: .212, eta 1: .560, eta 2: .422.

Post-warm-up (iteration >=200) pooled term Spearman .489, BSRS .441. Per-term results vary substantially and must be interpreted cautiously.

Across all terms within each run, correlations are about .94-.97 in baseline and no-term arms, about .15-.17 in BSRS eta .5 arms, and negative (~-.08 to -.09) in BSRS eta 2 arms. Such within-run correlations partly capture cross-term magnitude ordering; they do not establish timewise prediction for each individual term.

## Critic refit, base_s1
Recorded fixed-critic vs refit gradient cosine relative to original trainer direction:

| Iteration | feet_slide fixed | feet_slide refit | action_rate fixed | action_rate refit |
|---|---:|---:|---:|---:|
|50|.9679|.9318|.9961|.9669|
|200|.9433|.8827|.9956|.9399|
|500|.9755|.9086|.9983|.9369|
|1000|.9617|.8279|.9993|.8793|
|1999|.9318|.7600|.9990|.8559|

Refitting changes direction materially at these points, but the absence of a measured no-removal null-refit noise distribution means we cannot say how much is caused by removal rather than refit optimization.

## Decision status
Q0: BSRS reference association descriptive only; not validated update-level prediction.
Q1: fixed-critic covariance identity is algebraically testable; empirical term correlations recorded. No claim of general rho theorem.
Q2: some reward terms exhibit meaningful descriptive associations, others weak/undefined; criterion T1 cannot yet be passed because no actual alternative optimizer update and no matched clustered uncertainty analysis.
Q3: refit directions differ; null-refit control remains to be computed.
Q4: removing ordinary reward terms changes the objective; any same-critic rho analogue is a local update diagnostic, not a guarantee about final behavior.

## Next analyses needed
1. Calculate per-run/per-term Spearman and run-cluster uncertainty, with the same outcome for BSRS and terms; compare to simpler baselines (term share, gradient norm where appropriate).
2. Verify covariance-derived rho exactly reproduces stored term correlation and separately verify BSRS rho.
3. Inspect saved model/sample tensor schemas and determine feasibility of null critic refit offline; do not assume it was run.
4. If T1 requires actual counterfactual optimizer updates, identify missing optimizer state and perform an explicitly labeled offline replay or mark T1 untestable.
