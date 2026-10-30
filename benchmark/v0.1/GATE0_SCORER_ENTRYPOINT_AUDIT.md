# Gate 0 scorer entrypoint inspection (2026-10-27)

**Finding: BLOCKED — no implemented v0.1 benchmark scorer identified in the inspected branch tree.** The benchmark directory contains technical configuration/actuator audit scripts and an exclusion-register validator; these are not a benchmark detection/attribution/location scorer. Therefore a passing runtime metadata-leakage claim cannot be made.

## Related existing scorer, not interchangeable

`src/sim2gate/v2/phase2_score.py` is the *historical v2 Phase 2* scoring program. Its documented Stage 1 is run by a key holder and explicitly loads a key with `load_key(path)`; it constructs folds from arm labels and seeds. Its feature loader `load_table(path)` reads named feature columns. The program has historical labels `Control`, `R-slip`, `Phi-slip-mild`, `Phi-delay`. This is not evidence of a leak in the proposed v0.1 benchmark, but it is **not** an acceptable direct substitute for a blind held-out v0.1 scorer. Do not copy its key-loading workflow into the benchmark inference process.

## Implementation gate before leakage testing

1. Define separate **scorer** and **custodian/evaluator** processes, with disjoint inputs.
2. Scorer accepts only frozen, documented measurement features and opaque random identifiers; no intervention label, seed, training recipe, checkpoint path, reward coefficients, key, or metadata derived from them.
3. Custodian retains intervention assignments and lineage; computes metrics only after predictions and their SHA-256 commitment are recorded.
4. Add synthetic forbidden-field canaries and score-invariance tests, including output/log checks.
5. Run these tests against the *actual v0.1 scorer implementation* and archive machine-readable results. Do not mark them passed merely because the intended schema is documented.

**No Gate 0 pass, benchmark freeze, GPU profiling, or confirmatory training authorized.**
