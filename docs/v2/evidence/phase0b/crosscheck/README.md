# Phase 0B Q1 gradient cross-check, Oct 25, 2026

**Result: passed.** Two independent implementations computed the policy-gradient direction g(Â) =
∇θ mean(Â · log π_θ(a|s)) on phase 0A's 256-environment samples, from the saved actors, at the 6 grad-save
iterations of all 10 runs: 60 files, 756 directions (13 per file; 12 for the no-slip and no-action-rate runs, which
have 10 reward terms).

| | Vihaan | Sim2Gate |
|---|---|---|
| Code | ppo-shaping-diagnostics 0.3.2 (his own script), float64, CPU | `sim2gate.v2.crosscheck`, commit `24784f2`, float64, CPU; uses neither the recorder nor RSL-RL's classes |
| Output | `vihaan-crosscheck.tgz`, SHA-256 `aff9e6b4d826c5dfbf9cfe923e39d97e17289f7c32d238c903ed21cb508bcb3e` (recorded before Sim2Gate's side was run) | `crosscheck_sim2gate_MANIFEST.json` (per-file SHA-256) |
| Saw the other side first | No | No (code committed before his file arrived) |

**Comparison:** all 756 directions present on both sides, same names and parameter order; **minimum cosine
0.9999999999999996** (rule ≥ 0.999999). `crosscheck_result.json`.

**Sim2Gate's identity checks of the rebuilt actor** (all 60 files): std matches exactly; the recorded mean and std
reproduce the recorded log π to at most 3.6e-6; the saved weights reproduce the recorded mean to 6.2e-4 – 5.6e-3
(collection ran with reduced-precision GPU matrix products); the nearest other grad-save actor is off by 1.6 – 6.2,
so the saved actor is clearly the one that collected each sample. Vihaan found the same independently.

| File | SHA-256 |
|---|---|
| `crosscheck_result.json` | `2458e484373c551fb8449a28f9f2610cef1b2d26a05e1fedda7f49b4822ce13f` |
| `crosscheck_sim2gate_MANIFEST.json` | `c5bc088d3be8093253d5f742749844a0932880ed2707a4acb624c5785fa13a33` |

Both direction sets (about 250 MB each) are kept outside the repository, on the instance (`v2_runs/crosscheck_*`).
