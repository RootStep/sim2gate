# G2 stage S0: findings from attempts 1 and 2 (2026-10-30)

**Status (latest):** attempt 4 met every S0 exit criterion, and both completed runs were independently re-checked off the instance (see the end of this note). **Reviewer confirmation is requested to close S0.** The sections below record attempts 1 to 4 in order.

## Attempt 1: collector bug, fixed

The collector could not find the robot in the scene, because Isaac Lab 3 writes the robot path pattern as `/World/envs/env_[^/]+/Robot` and the collector handled only `env_.*`.

It was fixed in `2eb1b03`. Path resolution now matches the pattern as a regex against real prims, and the collector traverses instance proxies when collecting collision prims. The other runtime checks passed in both attempts:
- **Active terminations:** `time_out` only.
- **Active events:** `physics_material`, `reset_base` and `reset_robot_joints`, with no removed event active.
- **Robot shape friction readback:** 0.8 / 0.6 on all 128 × 27 shapes.
- **Motor model:** swept motor, lag 0, strength 1.0.
- **Delay buffers:** none.
- **Configured ground material:** 1.0 / 1.0, `multiply` mode.
- **Actor:** a non-recurrent MLP.

## Attempt 2: contact-material criterion not met

What the stage shows:

| | Value |
|---|---|
| Robot collision prims (env 0) | 27, the same as the simulator's 27 shapes per environment, so the traversal is complete |
| Robot collision prims with a bound physics material | **0**; all 27 are unbound |
| Ground collision prims | 1, bound to its own material `/World/ground/terrain/physicsMaterial`: 1.0 / 1.0, `multiply` |
| Scene default material | `/physicsScene/defaultMaterial`: 1.0 / 1.0, `multiply` |
| Robot friction at runtime (simulator readback) | 0.8 / 0.6 on every shape, set by the `physics_material` event |

The criterion in the manifest fails on "no unbound robot collision shape". The second listed problem, "no robot collision shapes found", is a reporting artifact: the check counted only bound shapes.

### Reading

- **The robot and ground do not share a material.** The ground has its own bound material. The robot's unbound shapes start on the scene default material, which is a different material.
- **The event sets the values but not the combine mode.** The `physics_material` event writes friction values (static, dynamic and restitution) through the simulator's tensor interface, and the readback confirms they took effect on every robot shape. That interface does not set combine modes.

  The robot shapes' runtime material therefore has one of two modes:
  - `multiply`, if the event modified the scene default material in place;
  - PhysX's default, `average`, if new materials were created.

  Which of the two happens is not shown by this evidence.
- **The effective friction is the same either way.** PhysX applies the higher-priority combine mode of the pair (average < min < multiply < max). The ground is `multiply` with 1.0, so with a robot material in either `average` or `multiply` mode the effective coefficient is robot × 1.0 = **0.8 / 0.6**, which is the cell. It would differ only if the robot material were in `max` mode. Nothing in the stack sets `max`, and no material in the scene is in `max` mode.

  This is an inference from documented PhysX rules and the recorded modes. It is not a direct measurement of the effective coefficient.

So the criterion did its job: it surfaced a configuration the check did not expect. The configuration does not appear to make the friction cells wrong. However, the reviewer asked for runtime evidence of the effective contact friction, and an inference falls short of that.

## Proposal for attempt 3 (not adopted; needs reviewer agreement)

Two changes.

**1. Revise the contact-material criterion.** Replace "no unbound robot collision shape" with the following:
- unbound robot shapes are allowed if the ground has its own bound material, distinct from the scene default;
- the effective coefficient is then computed for each robot combine mode the stack can produce: the scene default's mode and PhysX's default, `average`;
- every result must equal the cell.

Under this rule attempt 2 would give 0.8 / 0.6 and pass. The criterion would still fail if the ground used the default material, if the ground were not in `multiply` mode with 1.0, or if any scene material were in `max` mode. The reporting artifact would also be fixed.

**2. Add a physical friction probe from the run's own data.**
- **What it measures:** for foot samples in contact (normal force above 10 N) that are sliding (horizontal foot speed above 0.05 m/s), the ratio of tangential to normal contact force, |F_t| / F_n.
- **What it should equal:** Coulomb friction makes this ratio equal the effective dynamic coefficient while sliding: 0.6 in N080 and 0.30 in F040.
- **How it is recorded:** the collector stores the median and interquartile range of the ratio, and the number of sliding samples, in the header. This needs the full contact force vector, not only its vertical component.

The probe measures what the reviewer asked about: the friction the feet actually experience. The proposal is to report it in S0 and in every run.

Whether to also make it a pass/fail criterion is the reviewer's call. A tolerance would have to be set before attempt 3. Sliding contacts are few and noisy, and a sensor's net force can mix several contact points.

## Requested decisions

1. Accept change 1, keep the original strict criterion, or require something else, such as binding an explicit robot material in the evaluation configuration.
2. Accept change 2 as reported evidence only, or as a criterion with a stated tolerance.
3. After that, the manager authorizes S0 attempt 3. Attempts 1 and 2 each used a few minutes, which is within the 0.5 GPU-hour S0 cap.

**Option not proposed: binding an explicit material to the robot in the collector.** That would change the evaluated physics relative to the Phase 2 runs, which used the same task and robot asset and so, presumably, the same unbound setup. That could only be done in a new manifest revision.

## Reviewer decisions (2026-10-30) and implementation

1. **Revised criterion: accepted, with a condition.** The robot's runtime combine mode must be verified where possible; where it can't be observed, the limitation is recorded.

   Implemented in `contact_problems()`. The robot's combine mode is taken from the first source available:
   - a runtime observation;
   - the bound material's mode;
   - every mode the stack can produce for an unbound shape (the scene default's mode, and `average`).

   In the third case, the limitation is written into `runtime_checks.contact.limitations`, and the check fails if any material on the stage uses `max`.

   The collector looks for a combine-mode query on the simulator's tensor view and records what it finds. A runtime query is used if one exists. The reporting artifact ("no robot collision shapes found") is fixed.
2. **Friction probe: accepted as diagnostic evidence only.**
   - The `FrictionProbe` class records contact samples, sliding samples, the median, the interquartile range, the thresholds and the stated limitations under `friction_probe` in the header.
   - It never affects validity; the analysis does not read it, and a test checks this.
   - Probe failures are recorded, never fatal.
3. **Attempt 3 needs explicit manager authorization.** The revised check is committed first. All other S0 exit criteria are unchanged, including the 1000-step trajectory and terminal-sample checks.

The robot's physics materials are not changed.

## Attempt 3 (2026-10-30, tools `4a665cc`): data criteria passed; clean exit not yet shown

Every data criterion passed:
- the run completed, the footer verifies, and `read_run` reports no problems;
- 128,000 samples were checked, with every step-synchronization counter at 0;
- the episode counter was 0 after the collector's reset;
- every episode ended at step 1000, and all 128 resets were observed outside the recorded samples;
- there were no early ends;
- the runtime checks passed;
- the effective friction is 0.8 / 0.6.

The recorded limitation: the robot's runtime combine mode cannot be observed, because the tensor view has no query for it. Both modes the setup could produce give 0.8 / 0.6.

**Not passed: exit code 0.** After the output was written and verified, Isaac Sim's shutdown (`SimulationApp.close()`, called by the play script when its loop is interrupted) hung. A 2-minute watchdog then ended the process with an error.

The runbook's PASS/FAIL summary did not include the exit-code criterion, so its "S0 PASSED" line overstated the result.

The data are unaffected, because the file was complete and verified before the shutdown began.

**Fix:** the collector now ends the process itself, with its verified exit code, as soon as the output is fsynced and renamed:
- 0 for a completed, re-verified run that passed step synchronization;
- 1 for a run that is incomplete or does not verify;
- 3 for a step-synchronization failure.

Attempt 4 (proposed, needs authorization) re-runs S0 once to show the exit criterion.

**Friction probe: not informative in this build.** Isaac Lab reports that PhysX does not return the total contact force, so the contact sensor returns only the normal force. The tangential part is numerically zero: the median |F_t| / F_n was 1.3 × 10⁻⁷ over 153,842 sliding samples. The probe therefore says nothing about friction here, which is what reviewer decision 2 anticipated.

A real probe would need the sensor's friction-force output, if this Isaac Lab version offers one (for example a `track_friction_forces` option). That is not proposed now.

**Contact-force note for G1.** In this build `net_forces_w` is the normal contact force. On flat ground its z component equals the vertical load, so the truth library's stance detection (z force above 10 N) is valid. It must be re-checked before any non-flat terrain is used.

## Attempt 4 (2026-10-30 21:12 UTC, tools `d7e88a6`): every S0 exit criterion met

The collector exited with code 0 after 69 seconds, with no shutdown hang. The run file is `p0v5b_s101__N080.attempt4.jsonl.gz`: 7,998,478 bytes, 128 episodes.

| Criterion | Evidence |
|---|---|
| Exit 0; completed; footer verifies; `read_run` reports no problems | exit 0; `completed: true`; no problems |
| 1000 consecutive post-physics samples | 128,000 samples checked; `capture_length_not_consecutive` 0; `steps_without_physics_advance` 0; `stale_env_samples` 0; lengths 1000..1000 |
| First recorded state synchronized | `reset_episode_length_max` 0; `command_mismatch` 0, max command deviation 3e-8 |
| Final sample is the terminal transition | `termination_flag_mismatch` 0; `terminal_steps` [1000]; `unexpected_early_dones` 0 |
| No reset state in the trajectory | `done_without_reset_after_capture` 0; `state_changed_after_capture` 0, max change 0.0; `reset_moved_robot` 128 |
| Configuration | terminations `['time_out']` (only `base_contact` was removed); no removed event active; motor lag 0, strength 1.0; no delay buffers |
| Effective friction | [0.8] / [0.6]. Limitation: the robot's runtime combine mode is not observable; both producible modes give the same value |
| Provenance | checkpoint `14e03fab…` matches the manifest; installed `sim2gate` dependency files match the repository |

The friction probe was not informative, for the same reason as in attempt 3.

The runbook's PASS/FAIL printer crashed on a variable-name bug of its own, after the evidence had been printed. The criteria above are read from that evidence. The printer is fixed.

**Proposed status:** S0 is closable. Before asking the reviewer to close it, the attempt 3 and attempt 4 run files will be re-checked independently, off the instance.

Measured cost: 69 s per run. S1 is not authorized; its cap is re-derived from this measurement as part of its approval.

## Independent off-instance check of attempts 3 and 4

The run files were copied off the instance in `phase2_tier_r_smoke.tgz`, whose SHA-256 (`8bdfe408…ab04a3`) matches the custodian's checksum taken on the instance. They were checked with [s0_independent_check.py](s0_independent_check.py). The result is in [results/s0_independent_check_2026-10-30.json](results/s0_independent_check_2026-10-30.json).

| Check | Attempt 3 | Attempt 4 |
|---|---|---|
| `read_run`, against the manifest version the run used | no problems | no problems |
| Episodes; env IDs; samples per episode | 128; 0–127; 1000 each | same |
| Command vs the command set, recomputed from the data | within 5 × 10⁻⁶ (the 5-decimal rounding) | same |
| Identical consecutive samples (stale reads) | 0 | 0 |
| Largest base step per 20 ms; largest height step | 2.35 cm; 1.5 cm. No teleport, so no reset state is recorded | same |
| Header step-synchronization and runtime checks | ok | ok |

**Reproducibility.** The two runs used different collector versions; the changes were in the exit path and the header writer. The seed, policy, cell and commands were the same. All 128 trajectories are **bit-identical** across the two runs.

The smallest net displacements over an episode are 0.14–0.22 m. Those robots were commanded to turn while walking, and each completes roughly one circle in 20 s, so a small net displacement is expected. The step continuity above is what rules out a reset.

**Not done:** no failure, fall or tracking outcome was computed, because S0 output is never analysed.
