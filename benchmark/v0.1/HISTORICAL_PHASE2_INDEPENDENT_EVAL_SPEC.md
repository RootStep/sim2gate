# Diagnostic Benchmark v0.1 — historical Phase 2 evaluation specification (revision 2)

**Status: DEVELOPMENT DRAFT, revision 2 — 2026-10-30.** Commit this specification **before** any new evaluation trajectories. It describes a retrospective, post-results-seen development exercise, **not** a preregistered confirmatory benchmark. It authorizes no training, GPU evaluation, Check 2 rerun, protocol freeze, key unsealing or external publication.

Revision 2 changes:
- adds the three evaluation tiers (section 1);
- records the Check 2 physics confound ([CHECK2_PHYSICS_CONFOUND_EVIDENCE.md](CHECK2_PHYSICS_CONFOUND_EVIDENCE.md));
- updates gate G0 with checkpoint verification evidence;
- gives both baselines a common 20-robot healthy reference;
- requires Sim2Gate verdicts to be computed and committed before new trajectories;
- uses Clopper–Pearson intervals throughout;
- defines the delay conditions;
- makes the slip truth consequence-based.

## 1. Three evaluation tiers — never mix them

| Tier | What it is | Physics | Status of its results |
|---|---|---|---|
| **H. Historical** | Phase 2 measurements as recorded on Oct 25–26, 2026 | Nominal, Check 1 and Check 5 under the common nominal profile; **Check 2 under each arm's training physics** | A record of what the original system measured. Check 2 is confounded for Φ-slip-mild and Φ-delay |
| **R. Corrected retrospective** | The same 19 policies re-measured in one common evaluation task, plus independent symptom tests | Identical for all policies | **Post hoc, development only.** The cohort has already informed the benchmark design, so this cannot be independent validation |
| **F. Future independent** | New policies not used for any threshold, design choice or analysis decision | Identical for all policies | The only tier that can support generalization, false-alarm or confirmatory claims. Not part of this specification |

Every table and sentence must say which tier it reports.

## 2. Cohort and known selection

| Arm | Completed / assigned | Training intervention | Interpretation |
|---|---:|---|---|
| Healthy controls | 5 / 5 | Unmodified P0 v5b, seeds 101–105 | Also among the 20 clean robots that set the v2 limits |
| R-slip | 5 / 5 | `feet_slide` reward weight ×0.1, seeds 301–305 | Reward-side slip intervention |
| Φ-slip-mild | 5 / 5 | Training friction shifted to 0.3–0.6, seeds 306–310 | Historical physics-side slip intervention; **not benchmark D1** (narrowed randomization) |
| Φ-delay | 4 / 5 | No command delay in training; completed seeds 311, 312, 314, 315 | Delay-fragility intervention; s313 has no final checkpoint |

N = 19 completed policies, plus the incomplete s313, reported explicitly.

The arms were kept after earlier pilots because they showed symptoms while keeping nominal viability. Phase 2 behaviour and intervention outcomes have already been seen. Every design choice here is therefore **post hoc relative to earlier results**, even when its execution settings are fixed before new trajectories. All 19 are seen development data, never future held-out policies.

## 3. Gate G0: checkpoints and environment

**Bytes: verified.**
- 14 study checkpoints are verified in [phase2_checkpoint_byte_verification_2026-10-30.json](phase2_checkpoint_byte_verification_2026-10-30.json). The source is `unsealed.tar`, operator-attested as decrypted from `sealed_for_vihaan.tar.enc`, whose SHA-256 `60228f28…` matches `docs/v2/PHASE2_FREEZE.md`. Each digest equals the study's `state.json` `checkpoint.sha256` at its recorded run folder.
- 5 controls are verified in [phase1_checkpoint_byte_verification_2026-10-30.json](phase1_checkpoint_byte_verification_2026-10-30.json), on the Brev instance.
- No digest equals the empty-file SHA-256 (`e3b0c442…`).
- 7 of the 14 study files were reported at 983,989 bytes in the operator's verifier output. **Pending:** the byte sizes of all 14 study members, taken from the archive listing and added to the verification record.
- These records are both operator-run on 2026-10-30, using `verify_checkpoints.py` at `f63c55e`.

**Still open before any new trajectory:**
- copy the 14 study checkpoints to the evaluation machine, then re-hash them there;
- load each checkpoint under the recorded environment (Isaac Lab 3.0.0, Isaac Lab-Arena `2cc9e97`, rsl-rl-lib 5.4.1, sim2gate `e502dae`) and record the result;
- confirm checkpoint selection: the final `model_1999.pt` of each run, as in Phase 2.

**Stop rule G0:** no new trajectory until every item above is recorded. Missing files are missing, not guessed. Partial recovery supports at most a labelled subset feasibility study, never a 19-policy result.

## 4. Common evaluation conditions (tier R)

- **One task for all policies:** `Sim2Gate-Velocity-Flat-Go2-P0-v5b`, with every condition applied by explicit override and recorded in each report. **Never** use a policy's training task.
- Before any tier-R run, verify that every report records the same task, friction, delay, events, command set and engine. A report that differs is invalid and is rerun, not analysed.
- Confirm from the historical nominal, sweep and Newton reports that each records the same `profile_record`. This upgrades tier H's claim that those three were unaffected from a code reading to runtime evidence.

## 5. Independent behavioural truth (tier R)

Intervention assignment comes from the historical training configuration and is **not** a behavioural diagnosis. Truth is measured with **separate code and fresh episodes**, without consulting Sim2Gate verdicts or thresholds.

**Slip arms (R-slip, Φ-slip-mild).**
- **Primary truth is consequence-based:** falls, or tracking loss above a stated threshold, on low-friction ground in the common task. Use a friction distribution and command sequences different from Check 2's fixed command set.
- **Secondary:** slip magnitude from a separately implemented estimator using raw contact and foot kinematics. It must be unit-tested on synthetic stationary and sliding trajectories.
- Predefine before reading results: contact logic, velocity frame, aggregation, episode length, minimum valid contacts, missing-sensor policy, repeat rule and effect threshold.

**Delay arm (Φ-delay).**
- **Delay values:** values not on Check 1's frozen grid, within the same realistic envelope. Check 1's delay buffer is limited to 20 physics steps (100 ms, `isaaclab_conditions.py:47`). Values beyond it require a separately reviewed implementation and are reported apart.
- **Truth:** episode-level falls and tracking loss from a separately written detector, against controls at the same delays and on the same seeds.
- Predefine: delay values, command trajectories, fall criterion, tracking-loss threshold, episode counts and aggregation.

**Independence disclosure:** separate code and conditions reduce circularity, but both evaluations rely on simulated physics and may share low-level telemetry. Neither is physical-robot ground truth. Thresholds are development-only.

## 6. Baselines and diagnostic comparators

**Common healthy reference.** Both baselines use the **same 20 qualified P0 v5b robots** (p0v5b_s101–s105, s107–s111, s113–s122). These robots also set the v2 Check 2 limits. So the reference is consistent for both methods, but it **cannot** provide an independent false-alarm estimate for either. False-alarm figures from it are in-sample (section 7).

**Success-only baseline.**
- Measurement: nominal PhysX success under the common nominal profile. For the 5 controls, use the **Phase 2 re-measure**, which ran on the same code as the study robots (`docs/v2/PHASE2_PROTOCOL.md:122-124`). For the other 15 reference robots, use their Phase 1 nominal measurement, and state the version difference.
- Primary rule: flag a policy whose success is strictly below the minimum success among the 20 reference robots. Report the numeric threshold and its source.
- Sensitivity: the 5-control minimum, a leave-one-out reference minimum, and the historical 0.85 viability cutoff (descriptive only, because it helped select the arms). Never tune a threshold to detect flawed policies.
- "Success misses the flaw" is a **selected-cohort phenomenon**, not a frequency estimate.

**Sim2Gate comparators.**
- **H (historical):** compute verdicts from the recorded measurements using the frozen v2 rules:
  - Check 1 envelope rule;
  - Check 2 combined worst-metric ratio above 1.0541 (`docs/v2/evidence/layer1/check2_limit_v2.json`);
  - Check 5: Newton fall-free share below 80% or tracking error above 2× PhysX.
  
  Label Check 2 for Φ-slip-mild and Φ-delay "measured under training physics". **Commit this verdict table before any new trajectory.** It is CPU-only.
- **R (corrected):** rerun Check 2, and Checks 1 and 5 for parity, in the common task under the same frozen rules and limits, with no recalibration. Commit that table before the independent truth labels are computed. Label it post hoc corrected.

## 7. False-alarm context

- Controls s101–s105 are in-sample for the Check 2 limits. Their flags are reported descriptively only.
- Development context: the existing leave-one-out across the 20 reference robots, with limits from the other 19 (`src/sim2gate/v2/layer1_limits.py:222-232`). It reports Check 2 at 0/20. The Layer 1 README quotes a Wilson interval (0.00–0.16); this specification reports Clopper–Pearson (0.000–0.168). Reproduce it from the code before citing it.
- Check 1 and Check 5 false-alarm context is reported separately; Check 2's result does not cover the combined decision.
- A credible false-positive rate needs tier F: new healthy policies not used to set any threshold.

## 8. Reporting

One row per assigned policy, with an explicit `missing` row for s313:

`policy_id | arm | checkpoint_verified | nominal_success_n/N | success_only_flag | H_check1 | H_check2 (physics label) | H_check5 | R_check1 | R_check2 | R_check5 | slip_consequence_failure | slip_magnitude | delay_falls_n/N | delay_failure | cross_symptoms | notes`

- Aggregate by arm. Report completed and missing denominators, independently manifested symptoms, and success-only flags. Report Sim2Gate flags separately for tiers H and R. Include the cross-tabulation of **independent truth × success-only × Sim2Gate (R)**.
- Use Clopper–Pearson 95% intervals for per-arm proportions (for example, 0/5 has an upper bound of 0.522). They are descriptive for a small, selected cohort.
- Report continuous effect sizes against control ranges as well as flags. The independent training run is the unit.
- **No headline AUROC, sensitivity at 10% false-positive rate, or generalization claim.** Record N/A where a method's necessary evidence is absent.

## 9. Gates and stop rules

1. **G0 — bytes and environment:** section 3.
2. **G1 — evaluator independence:** the independent slip and delay code passes unit tests. Verify that it makes no calls to Sim2Gate scoring code and uses none of its thresholds.
3. **G2 — prospective manifest:** commit a separate manifest before any run. It fixes the common task and conditions, delay values, friction distribution, command sequences, episode budgets, seeds, failure definitions, missing-data rules, hardware, and the GPU-hour and cost cap.
4. **G3 — comparator lock:** the tier-H verdict table, the baseline thresholds and the reference set are committed before any tier-R trajectory. The tier-R Sim2Gate table is committed before truth labels are computed.
5. **G4 — completeness:** preserve every attempted evaluation, crash and invalid report. Never replace a policy because of its outcome.
6. **G5 — analysis:** report all tiers separately, including negative and ambiguous findings, in an internal development report.

**No new GPU run is approved.** Execution needs G0–G3 complete, a committed G2 manifest and explicit manager approval.

## 10. What this exercise cannot establish

It cannot validate a confirmatory benchmark. It cannot establish a population failure frequency, an unbiased false-alarm rate, mechanism-level attribution or hardware transfer. It tests:
- whether independent measurement is feasible;
- whether a known, selected phenomenon survives common evaluation physics and independent conditions;
- whether success-only and Sim2Gate each detect it.

Stronger claims need tier F.
