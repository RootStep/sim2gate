# G2 revision 2 — response to the reviewer's decision (2026-10-30)

**Status:** G2 provisionally accepted. The manager authorized S0 only, on 2026-10-30 (20:04 UTC). S1 and S2 are not authorized.

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
| 4 | Hard cap, cost estimate, staged approval; 133 runs not authorized | Stages S0 (1 run), S1 (40) and S2 (112). Each has its own approval fields, GPU-hour cap and dollar cap: 0.5 h / $1.75, 3.0 h / $10.50 and 7.5 h / $26.25, for a total of 11.0 h / $38.50 at the Brev billed rate of $3.50/h (confirmed by the custodian on 2026-10-30). Caps are never raised within a stage. The collector requires `--stage`. | manifest `stages`, `cost_model`, `total_caps`; `authorize()` |
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
2. Optionally approve S0 alone (1 run, cap 0.5 h / $1.75). The custodian confirmed the Brev billed rate of $3.50/h on 2026-10-30.
3. Choose 128 or 256 episodes per cell.

S1 and S2 are not requested now.

## Revision 2b: step synchronization and effective friction

The reviewer's additional finding was that the correspondence between the recorded samples and the environment's own step, termination and reset bookkeeping was asserted but not demonstrated in Isaac Lab. It also asked whether the friction readback measures the actual ground contact. Decision: G2 provisionally accepted; S0 pending explicit manager authorization.

### 1. Step synchronization is now measured on every run

At the capture point, inside reward computation, the collector records for every environment:
- the environment's episode length;
- its termination flag;
- its base position;
- its velocity command.

After the step returns, it records the episode length and base position again. `SyncTrace` checks each open environment at each step:

| Check | Fails if | Rules out |
|---|---|---|
| Capture-time episode length = policy step k | any sample is not k | skipped, repeated or off-by-one samples; desynchronized counters |
| Capture-time termination flag = the done flag the step returns | any mismatch | a snapshot taken at a different point than the terminations |
| Done: episode length after the step = 0 | not reset | treating a non-reset step as terminal |
| Not done: length and position unchanged after the capture | any change | anything (resets included) between capture and return |
| Captured command = the pinned command | any deviation over 1e-5 | command buffers not synchronized after the reset |
| Some environment moved since the previous capture | no environment moved at a step | stale, non-post-physics reads |
| Every environment ends at step `max_episode_length`, with the flag set | otherwise | episodes that do not end at the time limit |
| Episode length after the collector's own reset = 0 | otherwise | the first-call reset not taking effect |

The step also counts `reset_moved_robot`: done environments whose state changed between capture and return. That positive evidence shows the reset happened and is outside the recorded sample. In S0 it should equal 128.

The results go into the header as `sync_checks`. Any problem makes the collector exit 3. The file is kept as evidence, and `read_run` marks the run invalid.

**The first call.** It resets every environment and discards the action the policy computed from the pre-reset observation; that action is never applied. The policy's next action is computed from the reset observation. The checks above show that episode counters, commands and the first recorded sample (length 1) line up after that reset.

### 2. Friction is checked as effective robot–ground contact

The collector reads physics materials from the stage:
- the material bound to every ground collision prim, or the scene default material if none is bound;
- the material bound to each robot collision prim in env 0;

each with its friction combine mode.

It combines the robot's simulator-readback friction with each ground material under PhysX's rule: the higher-priority mode of average < min < multiply < max wins. The result must equal the cell.

Two further failure conditions:
- the robot and the ground share a material;
- a robot collision shape has no bound material, so it would fall back to the scene default.

Either fails verification.

The configured ground material (`scene.terrain.physics_material`, `sim.physics_material`) is also recorded from the config. A runtime-check failure writes `<out>.diagnostic.json` with everything observed, so S0 produces evidence even when it fails.

Cell F040 is valid only if the effective coefficient is 0.40. For example:
- with a ground of 1.0 in multiply mode, the effective coefficient is the robot's value, 0.40;
- with a ground of 1.0 in average mode, it would be 0.70, and the cell would be invalid.

The check makes no assumption about which case holds; S0 records it.

### 3. Tests

The synthetic suite now has 32 tests. The full benchmark suite has 117 tests, all passing. New tests cover:
- a clean trace;
- each of the six desynchronization types;
- a reset that did not take;
- an early end;
- an end-to-end step loop in which a fall on the terminal step is recorded and the reset is detected but not recorded;
- the combine rule;
- seven contact-material failure cases (average-mode ground, max-mode robot, ground not 1.0, no ground material, unbound robot shape, shared material, unreadable stage);
- invalidation of runs whose header lacks passing `sync_checks`.

**What S0 must still show.** The S0 exit criteria in the manifest list each value the smoke run must report. The open question is whether the Isaac Lab runtime behaves as these checks expect: reward computation after terminations and before resets, `reset_buf` at that point, and `episode_length_buf` reset to 0. The checks make no assumption about it. If the runtime differs, S0 fails loudly and shows how.

