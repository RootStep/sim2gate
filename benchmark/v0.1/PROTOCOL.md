# Diagnostic Benchmark v0.1 — protocol draft

**Status: DRAFT — requires approval and SHA-256 freeze before confirmatory training.**

## Scope
Unitree Go2 locomotion policies trained in Isaac Lab/RSL-RL. Test **Detect**, **Attribute**, and **Locate**. Do not infer hardware transfer from simulation.

## Population and labels
- Healthy controls and deliberately flawed policies; prior 23-policy set is seen and may be used only for development.
- The specification currently describes **five** flaw categories while calling them **six**: training, dynamics, simulator exploits, engine dependence, naturally occurring failures. **OPEN:** reconcile taxonomy.
- A planted intervention is ground truth for the *intervention*, not proof that a behavioural flaw manifested. Define independent manipulation checks and handling of failed manifestations.
- Naturally occurring failures may have unknown causes; predefine unknown/abstention scoring.
- Proposed held-out target: 40 flawed + 40 healthy policies, **conditional on compute pilot and power/precision assessment**.

## Experimental split
Separate seen development/calibration policies from fresh held-out policies by training run/seed and any shared recipe families as specified before generation. Do not allow test-set tuning. Keep random IDs and sealed answer key outside method inputs. Audit metadata leakage.

## Methods
1. Nominal success only
2. Randomized robustness aggregate
3. Individual raw training signals
4. Arena-style sensitivity analysis (define exact operational analogue and limits)
5. Frozen Sim2Gate diagnostic output

Use matched evaluation budgets or report differences; methods incapable of a task receive **N/A**, not a scored failure.

## Outcomes
- **Detect:** AUROC plus detection rate at a **proposed** calibration-set 10% false-alarm rate. Freeze score orientation, threshold, ties and missing-run rules.
- **Attribute:** accuracy on eligible known intervention/cause labels, with separate unknown/abstain reporting; do not assume an intervention uniquely establishes a mechanism.
- **Locate:** compare predicted failing disturbance cells to an independently measured dense reference grid; report cell-level precision, recall, F1 and per-policy aggregation.
- Report uncertainty at the **independent policy training run** level, not treating rollouts or cells as independent replicates. Include subgroup counts and costs.

## Exclusions and reproducibility
Define crash/retry/exclusion policy **before** new training. Preserve failures and protocol deviations. Hash source commit, protocols, recipes, checkpoints, evaluation grids, predictions, ground-truth key commitments and analysis scripts. Document software/hardware versions and external reproduction instructions.

## Sequence and gates
1. Evidence and threshold inventory, including applicability of P0 v4 versus P0 v5b limits.
2. Review and approve taxonomy, baselines, dense-grid truth, calibration, and power/compute.
3. CPU synthetic dry run including deliberately missing records and leakage tests.
4. Fresh **development pilot**: proposed two healthy + two flawed policies. No confirmatory claims.
5. Freeze code, labels, seeds, scoring and held-out recipe generation.
6. One-shot blinded evaluation and independently checked analysis.
7. Reproducibility package and publication.

**Do not alter the existing frozen v2 studies or reuse their thresholds without applicability checks.**
