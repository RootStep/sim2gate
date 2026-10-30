# G2 revision 2 — response to the reviewer's decision (2026-10-30)

**Status: resubmitted for review. The manifest stays DRAFT, and nothing is authorized to run.**

The reviewer's decision:
- G1: provisional.
- G2: revision required.
- GPU smoke test: not authorized.
- Full evaluation: not authorized.

This note maps each item to the change made.

Files changed:
- [G2_TIER_R_MANIFEST.json](G2_TIER_R_MANIFEST.json) and [.md](G2_TIER_R_MANIFEST.md)
- [independent_analysis.py](independent_analysis.py)
- [independent_collector.py](independent_collector.py)
- [tests/test_independent_pipeline.py](tests/test_independent_pipeline.py)

The thresholds and acceptance criteria are unchanged: `truth_rules`, the alpha level and the 0.10 floor are the G1 proposals. They are now marked provisional.

## Reviewer items

| # | Reviewer request | Change | Where |
|---|---|---|---|
| 1 | Keep observation latency as a distinct experiment, not equivalent to Check 1 | O1 and O2 (20 and 40 ms observation delay) are secondary generalization cells. They are listed in `secondary_cells` and never decide truth. A test checks that primary and secondary cells are disjoint. | manifest `cells`, `secondary_cells` |
| 1b | Prefer action/command delay as primary | A1 and A2 (20 and 40 ms action delay at the policy interface) are Φ-delay's primary cells. The 60 ms observation cell L3 is dropped. Justification below. | manifest `primary_cells`; collector `act_buf` |
| 2 | Thresholds provisional, for smoke testing only | `thresholds_status`: usable for S0 only. They must be confirmed, or changed in a new revision, before S1 is approved. No change is allowed after S1 starts. | manifest `thresholds_status`, `stop_rules` |
| 3 | 128 episodes initially; review power | 128 is kept. An exact power table is added: power is at least 0.90 at +0.15 and about 0.50 at +0.10. "Not manifested" is defined as "no excess of about 0.15 or more detected". 256 episodes remains an open option. A test re-derives one table entry. | manifest `power` |
| 4 | Hard cap, cost estimate, staged approval; 133 runs not authorized | Stages S0 (1 run), S1 (40) and S2 (112). Each has its own approval fields, GPU-hour cap and dollar cap: 0.5 h / $1.75, 3.0 h / $10.50 and 7.5 h / $26.25, for a total of 11.0 h / $38.50 at a planning rate of $3.50/h. Caps are never raised within a stage. The collector requires `--stage`. | manifest `stages`, `cost_model`, `total_caps`; `authorize()` |
| A | Invalid runs still influence the analysis | `read_run` now checks completion, footer, manifest, checkpoint digest, cell, command set, task, `num_envs`, runtime checks, early dones, exact episode length and schema. Any problem makes the run invalid, and it is excluded everywhere, including from the pooled controls. Invalid and missing runs are reported separately. If a required primary or control run is missing or invalid, the result is `INCOMPLETE`, no truth labels are issued and the program exits 2. | `independent_analysis.py` |
| B1 | Runtime verification of the effective configuration | Before the first recorded step, the collector reads back, from the simulator, the active terminations, the active events, per-shape static and dynamic friction, the motor model type, lag and strength, and which delay buffer is engaged. Any mismatch aborts the run. The readback is stored in the header, so the analysis can reject runs that lack it. | `_verify_runtime`, `runtime_problems()` |
| B2 | Terminal transitions must not conceal falls | State is captured inside reward computation. That is after physics and terminations and before the environment resets ended episodes, the same capture point as the existing condition runner. The terminal sample is recorded and then the episode is closed. Any episode ending before the time limit is counted, and that invalidates the run. | `_install_prereset_hook`, `EpisodeLog.add` |
| B3 | Explicit completion, not KeyboardInterrupt | Output goes to a temporary file. It is renamed only after every episode is written, with header `completed: true` and a footer holding the record count and SHA-256. `main()` returns 0 only if the run completed **and** the re-read file verifies. KeyboardInterrupt remains only as the signal that stops Isaac Lab's play loop; it is never taken as success. | `EpisodeLog.write`, `verify_written`, `main` |

## Why action delay is primary, and what it does not establish

Φ-delay was trained without actuator command delay. The other arms trained on P0 v5b physics, which includes 0–30 ms command delay. The flaw therefore lives in the command channel, and the primary truth cell delays that channel.

The implementation is separate from Check 1:
- Check 1 delays the joint target inside the swept DC motor model, in 5 ms physics steps.
- The collector holds the policy's action for whole 20 ms control steps before the action term.
- The run asserts that the motor model's lag is exactly zero, so the two never stack.

Outcome labels come from the independent truth library, on a different command set.

**Limitation.** At 20 and 40 ms the perturbation is physically close to Check 1's grid points of the same values. A1 and A2 are therefore an independent implementation and judgment of the same kind of perturbation, not an independent kind of perturbation. Agreement with Check 1 on Φ-delay is partly expected by construction, and the report must say so.

O1 and O2 delay a different channel. They are reported as a secondary generalization test with no truth weight.

## Tests

The synthetic suite (no Isaac Lab, no torch) has 27 tests, up from 12. The full benchmark suite has 112 tests, all passing. The new tests cover:
- every invalidity reason, plus truncated, tampered and corrupt files;
- an invalid control run excluded from the pool, and the INCOMPLETE status that follows;
- missing primary vs missing secondary runs;
- exit codes 0 and 2 of the analysis;
- terminal-sample recording, including a fall at the terminal step that a reset would otherwise hide;
- early-done counting;
- the missing-capture guard;
- every runtime mismatch type;
- stage authorization;
- the stage partition and cap arithmetic;
- the delay-cell roles;
- one power table value.

**What is not yet tested.** The Isaac-facing reads cannot be tested here: termination and event manager attributes, `get_material_properties`, the motor attributes, and whether reward computation runs inside `step`. They are exactly what S0 is for. Each fails loudly rather than silently: a missing attribute raises, and a missing capture raises.

## Requested decision

1. Approve the manifest revision, or return it for further changes.
2. Optionally approve S0 alone (1 run, cap 0.5 h / $1.75), after the custodian confirms the Brev billed rate.
3. Choose 128 or 256 episodes per cell.

S1 and S2 are not requested now.
