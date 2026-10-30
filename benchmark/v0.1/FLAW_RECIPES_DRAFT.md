# Diagnostic Benchmark v0.1 — flaw recipes and ground-truth rules

**Status: DRAFT FOR TECHNICAL REVIEW — 2026-10-27. Not approved, frozen, or executed.** Companion to PROTOCOL.md and PROTOCOL_DECISIONS.md. No thresholds in this document are claimed validated.

## Label model: do not collapse these fields

For each policy record: `policy_id`, `training_run_id`, `parent_recipe_id`, `intervention_id`, `intervention_family`, `intervention_parameters`, `intervention_applied` (verified from immutable config/log), `manifestation_status` (present/absent/indeterminate), `manifestation_metrics`, `known_cause_status`, `diagnostic_eligibility`, `checkpoint_sha256`, `config_sha256`, and `evidence_refs`. Sealed fields must not be exposed to diagnostic methods.

**Intervention truth** means the intended change was actually applied. **Behavioral truth** means a separately defined, reproducible failure was measured. **Mechanism truth** requires additional causal controls, not just temporal association. Report all three independently. A planted-but-nonmanifesting policy must not be quietly dropped or relabeled as a healthy control.

## Candidate controlled recipes (not final count)

| ID | Family | Candidate intervention | Independent manipulation check | Independent manifestation test | Causal caveat |
|---|---|---|---|---|---|
| T1 | Training | Remove or strongly reduce slip-related penalty in reward recipe | Reward config diff and per-term contribution logs | Sealed contact/slip test on standardized terrain and commands; compare sustained slip and slip p95 | Absence of a penalty need not cause slip; check stock-task baseline and prevent a no-op intervention |
| T2 | Training | Modify reward weights to favor speed tracking over stability | Frozen reward-weight diff, reward term audit | Held-out perturbation episodes: tracking versus falls/tilt/contacts | Multi-objective tradeoff, not proof of unique cause |
| D1 | Dynamics | Train with intentionally narrowed friction randomization range | Training domain-randomization config and sampled friction histogram | Evaluate on friction levels outside the training range, with matched nominal control | A narrower range may not reduce robustness; do not assume manifestation |
| D2 | Dynamics | Train without a defined actuator delay or motor-strength variation | Logged training parameter distributions | Held-out delay/weak-motor grid with fall/tracking metrics | Distinguish missing training variation from deployment perturbation |
| S1 | Simulator exploit | Candidate contact/reward exploit encouraging nonphysical foot-ground behavior | Config and trajectory evidence of exploit opportunity | Independent contact-consistency and force/penetration audit, plus alternate physics settings | Must first demonstrate exploit exists; avoid creating an unphysical failure solely in the evaluator |
| E1 | Engine dependence | Train in one backend with a controlled backend-specific setting | Simulator/backend and config fingerprint; controlled replay settings | Matched PhysX-versus-Newton evaluation of same checkpoint and command distribution | Engine mismatch is a behavioral property, not automatically an intentional training defect |
| N1 | Naturally occurring | No planted intervention; prospectively identify spontaneous failure in new policies | Immutable absence of planted recipe; blinded failure discovery | Independent sealed disturbance-grid failure evidence | Cause may be unknown; exclude from known-cause attribution denominator, include in separate detection analysis |

**Controls H0:** standard training recipe with no planted intervention. Apply predeclared nominal qualification and report all training/qualification failures. Healthy means qualified under a specified envelope, not immune to all disturbances. Do not select healthy policies after seeing Sim2Gate output.

## Ground-truth design and leakage prevention

1. Use an evaluator independent of Sim2Gate scoring thresholds. Predeclare commands, disturbances, simulator settings, rollout seeds, episode count and measurement rules. If both tools rely on the same raw physics telemetry, disclose shared measurement dependence and keep labels derived independently of diagnostic predictions.
2. Define a reference cell as failed using a precommitted physical/behavioral criterion (e.g., fall, severe sustained slip, tracking loss) and a minimum repeatability rule; exact numeric cutoffs require pilot validation and are **OPEN**.
3. Separate manipulation checks (intervention applied), manifestation checks (observed behavior), and attribution checks (causal mechanism). Do not infer causal attribution solely from correlation or recipe name.
4. Generate paired controls using matched training setup and predeclared seeds where feasible; keep independent training runs as the inferential unit.
5. Preserve failures to train, nonmanifesting interventions, timeouts, missing telemetry, and duplicate/retried runs with explicit reason codes and a predeclared denominator.
6. Seal intervention identities and truth-grid outcomes from all methods until predictions are committed; audit checkpoint paths, run names, JSON keys, filenames, reward-term names, metadata and train logs for label leakage.
7. Avoid historical threshold reuse: P0 v4 and P0 v5b thresholds are not benchmark v0.1 truth criteria.

## Primary-analysis population — open decision

Suggested **two estimands** rather than one ambiguous 'flawed' label:
- **Intention-to-intervene:** detect assignment to a planted recipe vs healthy-control assignment, regardless of whether the defect manifests. This tests intervention recognition but may penalize correctly finding no actual problem.
- **Manifestation-based:** detect independently demonstrated behavioral failure vs qualified healthy controls, with indeterminate cases and selection effects explicitly reported. This tests diagnostic usefulness but conditioning on manifestation can bias comparisons.

Neither estimand should be presented as a proven causal diagnosis. Report family-level results and an exploratory naturally occurring failure cohort separately. Select the primary estimand **before** confirmatory training.

## Reviewer decision checklist

- [ ] Confirm each intervention is a real code/config change, not a no-op in the current Go2 task.
- [ ] Approve intervention strength levels using development-only runs.
- [ ] Predefine independent failure thresholds, reference grid and repeatability criteria.
- [ ] Decide whether engine dependence is a flaw label, a measured phenotype, or both.
- [ ] Decide which recipes permit defensible causal attribution.
- [ ] Lock healthy qualification and handling of nonmanifesting planted policies.
- [ ] Specify baseline input parity and prediction/truth schemas.
- [ ] Estimate compute and uncertainty before choosing confirmatory cohort size.

**No held-out runs may start on this draft.**
