# Sim2Gate Diagnostic Benchmark v0.1

**Status: draft / not frozen.** This is the scientific evaluation of **Sim2Gate**, not a second product.

## Questions
1. **Detect:** Can a method distinguish healthy from flawed Go2 policies?
2. **Attribute:** When evidence supports it, can it identify the known planted cause family?
3. **Locate:** Can it identify the disturbance conditions where a policy fails?

Fix verification belongs to separate v2 Phase 3 research and is **out of scope**.

## Evaluation approach
Compare nominal success, randomized robustness, raw training diagnostics, Arena-style sensitivity analysis, and a frozen Sim2Gate method. Evaluate each only on tasks it can meaningfully perform. Keep the earlier 23-policy blinded set as **seen development data**. Any confirmatory test must use independently generated, fresh held-out policies.

Read [PROTOCOL.md](PROTOCOL.md) before implementing. [FREEZE.md](FREEZE.md) is intentionally unapproved; no held-out training or scoring is authorized by this branch.

## Directories
- `configs/`: versioned cohort and method settings (draft placeholders)
- `baselines/`: baseline specifications and, later, adapters
- `results/`: schema and reproducibility notes, not hidden labels or bulky checkpoints

Existing product code remains in `src/sim2gate/`; existing frozen studies remain in `docs/v2/`.
