# Diagnostic Benchmark v0.1 — historical Phase 2 independent evaluation specification

**Status: DEVELOPMENT DRAFT — 2026-10-30.** Commit this specification **before** any new evaluation trajectories. This is a retrospective, post-results-seen development exercise, **not** a preregistered confirmatory benchmark. No training, GPU evaluation, protocol freeze, key unsealing, or external publication is authorized by this file.

## 0. Checkpoint availability: mandatory first gate

Fresh independent trajectories require the **actual trained checkpoint bytes**, not just recorded SHA-256 strings or Phase 2 feature tables.

- Five historical controls: `p0v5b_s101`–`s105`, reported in `phase1-clean.tgz`.
- Fourteen completed study runs: R-slip `s301`–`s305`, Φ-slip-mild `s306`–`s310`, Φ-delay `s311`, `s312`, `s314`, `s315`. The Phase 2 raw export reportedly contained 21 folders and 426 checkpoint files, but its current location and availability have **not** been verified here.
- `s313` crashed twice and has **no completed measurement**; do not substitute its training features or impute a result.

**Stop rule G0:** Do not schedule new evaluation until all required checkpoint bytes are located, per-run checkpoint selection is reconciled to the historical protocol, SHA-256 and lineage are checked, and the selected checkpoints can load under a documented compatible environment. Missing files are **missing**, not guessed. If G0 fails, permit only a separately labeled retrospective analysis of already recorded measurements; that analysis cannot establish independence through new terrain/command/delay trials. Partial recovery may support a transparent *subset feasibility study*, never a 19-run result.

## 1. Historical population and known selection

| Arm | Completed / assigned | Historical training intervention | Interpretation |
|---|---:|---|---|
| Healthy controls | 5 / 5 | Unmodified P0 v5b, seeds 101–105 | Controls were among the 20 clean robots used to set v2 limits |
| R-slip | 5 / 5 | `feet_slide` reward weight ×0.1, seeds 301–305 | Reward-side slip intervention |
| Φ-slip-mild | 5 / 5 | Training friction shifted to 0.3–0.6, seeds 306–310 | Historical physics-side slip intervention; **NOT benchmark D1** (narrowed randomization) |
| Φ-delay | 4 / 5 | No command delay in training, completed seeds 311,312,314,315 | Delay-fragility intervention; s313 missing |

**N=19 completed policies**, plus the incomplete assigned s313 explicitly reported. This cohort was selected after earlier pilots showed symptoms while preserving nominal viability. Prior Phase 2 Check 1/2 behavior and intervention outcomes have already been seen by the team. Independent-test design is therefore **post hoc relative to earlier results**, even if its new execution settings are fixed before new trajectories. All runs are seen development data, never future held-out policies.

## 2. Separate intervention truth from independent behavioral truth

Intervention assignment is established from historical training config/manifest and is not a behavioral diagnosis. A symptom must be measured with a **separate implementation and fresh evaluation episodes**, without consulting Sim2Gate check verdicts or thresholds to label it.

**Slip test (proposed, requires implementation review):**
- A separately implemented slip estimator from raw contact, body/foot kinematics and simulator state; independent unit tests with synthetic stationary/sliding trajectories.
- Distinct terrain friction distribution, command sequences, episode seeds and condition cells from Check 2's calibrated fixed command set and historical measurement suite.
- Predefine foot contact logic, velocity frame, slip metric aggregation, episode duration, minimum valid contacts, missing sensor policy, repeatability rule and absolute/relative effect threshold **before reading new results**.
- Report continuous slip metrics and per-policy differences from matched controls. Do not define a failure by Check 2's `slip_p95` cutoff or by its `flagged` output.

**Delay test (proposed, requires implementation review):**
- Evaluate at actuator/command delays **outside** Check 1's frozen delay grid, with a separately written episode-level fall detector and tracking-loss metric.
- Predefine delay values, command trajectories, fall criterion, tracking-loss threshold, episode counts and how instability is aggregated.
- Compare Φ-delay with controls using fresh episode seeds. Do not use Check 1's `action_latency_ms` flag to define truth.

**Independence disclosure:** Different code and conditions reduce circularity, but both evaluations still rely on simulated physics and may share low-level telemetry. They are not physical-robot ground truth. New thresholds are development-only and may be revised only in a newly versioned, explicitly post hoc analysis.

## 3. Nominal-success-only baseline

- Extract each policy's historical **nominal PhysX success rate** with its episode count, success definition and measurement-code version. Do not silently mix unlike success definitions.
- Report the original historical 0.85 viability cutoff as a **descriptive comparator only**; it helped select the arms and is not an unbiased detection threshold.
- Primary development success-only rule: define a fixed threshold from the **healthy reference distribution** before viewing the new independent evaluations, e.g. the minimum nominal success among the five historical controls; flag a policy if its success is **strictly below** that value. State ties explicitly and report the numerical threshold and source. This is an in-sample descriptive baseline, not a calibrated false-positive-rate rule.
- As a sensitivity analysis, also report the original 0.85 rule and a leave-one-out healthy-control threshold, with the reference control excluded when scoring itself. Do not optimize a threshold to maximize flawed-policy detection.
- The statement “success misses the flaw” is a **selected-cohort phenomenon**, not an estimate of its population frequency, because interventions with poor nominal success were screened out in prior pilots.

## 4. Sim2Gate outputs and false alarms

- Preserve the **historically recorded** Check 1/2/5 predictions as the development comparator, with scoring version, calibration and applicability noted. Do not recalibrate after seeing the independent outcomes.
- **No independent false-alarm estimate from controls s101–s105:** they are part of the 20 clean robots used to derive v2 Check 2 thresholds. Report their in-sample flags descriptively only.
- For development false-alarm context, use **leave-one-out across all 20 healthy calibration robots** with calibration re-fit on the other 19 for each held-out robot. The existing `docs/v2/evidence/layer1/README.md` reports Check 2 **0/20** under this procedure, with an approximate 95% interval of **0.00–0.16**; verify the underlying code and records before independently reproducing that claim. This is still not a truly untouched healthy cohort.
- Report Check 1 and Check 5 false-alarm context separately; do not apply Check 2's leave-one-out result to the combined Sim2Gate decision.
- A credible fresh false-positive-rate estimate requires **new healthy policies not used to set thresholds**.

## 5. Comparison table and descriptive statistics

Produce one row per assigned policy, with explicit `missing` for s313:

`policy_id | arm | checkpoint_verified | nominal_success_n/N | success_only_flag | check1_status/flag | check2_status/flag | check5_status/flag | independent_slip_metric | independent_slip_failure | independent_delay_fall_n/N | independent_delay_failure | cross_symptoms | notes`

Aggregate by arm:
- Completed and missing denominators; intervention application verified; independently manifested symptom count; nominal-success flags; Sim2Gate flags; joint cross-tab **independent manifestation × success-only × Sim2Gate**.
- Exact binomial (Clopper–Pearson) 95% confidence intervals for per-arm counts/proportions; for **0/5**, the upper bound is about **0.522** and for **0/20**, about **0.168**. Treat intervals as descriptive for this selected, small cohort, not as population-generalizable uncertainty.
- Report continuous effect sizes and control ranges, not only binary flags. Policy training run is the unit; do not treat episodes or disturbance cells as independent policies.
- **No headline AUROC, sensitivity-at-10%-FPR, or generalization claim** from 19 selected development runs. Do not score an approach as wrong when its necessary evidence is absent; record N/A or missing.

## 6. Execution and stop rules

1. **G0: bytes and environment.** Recover all 19 selected checkpoint files and verify SHA-256, selection lineage and loadability. Stop new-trajectory evaluation if not satisfied.
2. **G1: evaluator independence.** Independent slip and delay implementations pass unit tests; confirm no calls to Sim2Gate scoring code or use of its thresholds in behavioral labeling.
3. **G2: prospective execution manifest.** Fix terrain and command distributions, out-of-grid delay values, episode budgets, random seeds, failure definitions, missing-data rules, hardware/runtime and GPU-hour/cost cap in a separately committed manifest **before** running new evaluation.
4. **G3: baseline lock.** Fix nominal-success threshold, ties, version, historical Sim2Gate output source and 20-control leave-one-out procedure before inspecting new trajectories.
5. **G4: data completeness.** Preserve all attempted evaluations, crashes, missing checkpoints and invalid telemetry; do not replace a policy because its outcome is inconvenient. Report deviations and missing denominators.
6. **G5: analysis.** Independently recompute truth labels from fresh trajectories, compare with locked diagnostic outputs, report arm-level counts and exact intervals, and publish negative/ambiguous findings in the internal development report.

**No new GPU run is approved.** An approved execution manifest and explicit manager approval are required after G0–G3. No fresh training is proposed. If archives cannot be recovered, produce an evidence-availability report and stop; do not claim an independent benchmark result.

## 7. What this exercise cannot establish

This selected cohort cannot validate a confirmatory benchmark, a population failure frequency, an unbiased 10% false-alarm rate, mechanism-level causal attribution, or real-hardware transfer. Its purpose is to test independent measurement feasibility and whether a *known selected phenomenon* survives different evaluation conditions, while documenting whether success-only and Sim2Gate detect it.
