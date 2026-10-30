# G2 stage S0: findings from attempts 1 and 2 (2026-10-30)

**Status:** S0 is not passed. Both attempts stopped at runtime verification before any step was recorded, and no run output was written. No scientific run has taken place. This note asks the reviewer to decide on one verification criterion before attempt 3.

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
