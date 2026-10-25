# v2 phase 0B: freeze record

| | |
|---|---|
| Frozen file | `docs/v2/PHASE0B_PROTOCOL.md` |
| SHA-256 | `239124e256207f942782175c0525c4367cfb1ed2d1557e8b0f5d4d70521e7330` |
| Decided by | VJ, Oct 18, 2026 (T1 margin 0.1; T2 PPO-noise yardstick; offline null refit added; all deep iterations, warm-up-excluded check reported) |
| Analyses run on phase 0A data before this commit | None. Only the outcome measurements (slip, jitter, walking) in `evidence/phase0a/README.md` had been looked at; the protocol does not test them. |

Check: `sha256sum docs/v2/PHASE0B_PROTOCOL.md` must print the hash above. Any later change to the protocol is an
amendment, listed below with its date, reason and VJ's approval, and must come before the update-level quantities
are computed.

## Amendments

- Oct 18, 2026, minutes after the freeze and before any analysis: a cross-reference in 4.1 corrected ("see 4.3" to "see 4.5"); the hash above is the corrected file's. First frozen hash was 500f1c9ec8ceb0cade371fbbb75ac66dff5897bc16d7069724226110fda982a3.
- **Amendment 2, Oct 18, 2026, 22:48 ET, approved by VJ, before any update-level quantity was computed.**

  **Who runs it.** VJ asked Claude to run the phase 0B analysis and report the results. It runs with Sim2Gate's
  `sim2gate.v2.phase0b`, on the fingerprinted bundle (SHA-256 a8938401…). Vihaan reviews the results, and two items
  stay his:
  - Q1's derivation and the cross-implementation gradient check (his code against Sim2Gate's, on the 256-environment
    sample);
  - the Q4 statement.

  Until he signs off, the answer is **provisional**.

  **Operational details, fixed now:**

  1. **Q0 (BSRS reference).** One pair per run, deep iteration and η ∈ {0.5, 1, 2}:
     (`bsrs_what_if_rho[η]`, `direction_cos_trainer["bsrs:η"]`). Use the six PPO runs (base, no_slip,
     no_action_rate), where the trainer's direction is g(Â). Q0 passes if Spearman ≥ 0.7 and the lower bound of the
     95% bootstrap interval is > 0.5. In the two BSRS arms, (training-η ρ, `direction_cos_trainer["A"]`) is reported
     as descriptive.
  2. **Q2.** One pair per run, deep iteration and reward term: (`corr_A_A_minus_term[k]`,
     `direction_cos_trainer["minus:k"]`). Use the same six PPO runs. The BSRS arms are excluded because there the
     trainer's direction is not g(Â).
  3. **T1.** S_k − S_ref is the difference in Spearman between Q2 and Q0, bootstrapped over runs with 10,000
     resamples. It passes if the 95% lower bound is > −0.1. It fails if the upper bound is < −0.1. Otherwise it is
     inconclusive.
  4. **Baselines.** These are compared by **absolute** Spearman, |S_ρ| − |S_b|, because some baselines move
     opposite to the update change:
     - `term_abs_share[k]`;
     - Var(T_k)/Var(A) from `moments`;
     - β_k = Cov(T_k, A)/Var(A) from `moments`.

     ρ beats a baseline if the 95% lower bound is > 0. ρ fails if any upper bound is < 0. Otherwise the baseline
     comparison is inconclusive.
  5. **T2.** At each of the 10 refit points (`base_s1`, terms `feet_slide` and `action_rate_l2`, iterations 50, 200,
     500, 1000 and 1999), compare cos(`refit_minus:k`, `minus:k`) from the saved direction vectors with the
     yardstick, the median of `minibatch_grad_cos_trainer` at that iteration.
  6. **Warm-up check.** Repeat steps 1–4 without iterations 0–49. A different answer makes the result
     inconclusive.
  7. **Q1 numerics.** These are computed on the sample dumps of every run's deep iterations, with relative error
     max|x − ref| / max|ref|:
     - A recomputed against the trainer's advantages;
     - the reward terms summed against the reward;
     - the closed-form ρ_k against the directly computed corr(A, A − T_k).

     Each passes at ≤ 1e-5. The offline null refit follows section 4.4, with seed 0.


## Log

- Oct 18, 2026, 23:06 ET: VJ shared a separately run exploratory analysis (`PHASE0B_INITIAL_REPORT.md`,
  `bsrs_reference.csv`, `term_diagnostics.csv`; SHA-256 13fbc4f9…, d277b697…, 827ebf86…). It was not produced by
  `sim2gate.v2.phase0b`. That code was committed at `12dfca1` (22:56 ET), before these files were received, and it is
  not changed in response to them. The two analyses are to be compared after `sim2gate.v2.phase0b` has run on the
  bundle.
- Oct 25, 2026: **decision recorded (VJ, in place of Vihaan's pending personal sign-offs).** VJ authorised a
  documented review of this protocol, the results README and the report to stand in for Vihaan's Q1 and Q4
  sign-offs. The protocol file reviewed matched its SHA-256 above.
  - Q1: the per-term formula is approved for centred A and T_k with nonzero variance, with the reading recorded in
    `PHASE2_PROTOCOL.md` section 6 (β > 1 is negative advantage correlation, not a reversed update; R = 0 gives ±1;
    undefined when constant; the fixed critic is not retraining; identical masks, discounting and time-out bootstrap).
  - Outcome wording: phase 0B **passed its recorded decision rules**. No unqualified claim that ρ reliably tracks
    ordinary reward terms. The limits stay attached to the result: ρ beat Var(T_k)/Var(A) by only 0.002; within-run
    tracking of `feet_slide` was 0.053; the critic-refit yardstick was weak; gradient directions, not replayed Adam
    updates, were measured.
  - Decision: ρ enters phase 2 as **a candidate diagnostic whose added value remains to be tested** (C3).
  - Still open: the independent gradient cross-check on the 256-environment sample (Q1's numerical rule), and the
    original Oct 9 gate report. The frozen result is not rescored.
- Oct 25, 2026: **gradient cross-check, Vihaan's side computed** (ppo-shaping-diagnostics 0.3.2, float64, CPU),
  `vihaan-crosscheck.tgz` SHA-256 `aff9e6b4d826c5dfbf9cfe923e39d97e17289f7c32d238c903ed21cb508bcb3e`, 60 files. He did
  not look at Sim2Gate's code or output. Two deviations from the steps Claude wrote, both errors in those steps:
  - **Log-prob check.** The steps asked for log π of the rebuilt actor to match the recorded value to about 1e-5. It
    matched only to 2e-3 – 2.6e-2, because the saved weights reproduce the recorded Gaussian mean only to
    6e-4 – 5.6e-3. His diagnosis: collection ran on the GPU with reduced-precision (TF32) matrix products. Evidence:
    the recorded mean and std reproduce the recorded log π to 3.6e-6 (the density is right); the std matches exactly;
    a neighbouring iteration's actor is off by about 1.7 in the mean. The directions are defined at the saved actor
    and do not use the recorded log π. Sim2Gate's own identity check was changed accordingly (commit `24784f2`)
    before its reference was computed; the cosine ≥ 0.999999 criterion is unchanged.
  - **756 directions, not 780.** The no-slip and no-action-rate runs have 10 reward terms, so their files hold 12
    variants. Each side follows each sample's own term list.
- Oct 25, 2026: **gradient cross-check PASSED.** On `sim2gate-arena-2`, code `24784f2`: Vihaan's
  `vihaan-crosscheck.tgz` matched its recorded SHA-256 (`aff9e6b4…`) and its own checksum list; Sim2Gate's standalone
  implementation (`sim2gate.v2.crosscheck`, which uses neither the recorder nor RSL-RL's classes) computed the same
  60 files. All **756 directions** agree, **minimum cosine 0.9999999999999996** (rule: ≥ 0.999999). Q1's numerical
  rule is met by two independent implementations.
