# Gate 6 — Live Go2 actuator integration: execution plan

**Status: PREPARED / NOT EXECUTED.** This is a mandatory pretraining gate for the approved H0×1/T1×1/D1×1/D2×1 *planning allocation*. It is **not** authorization for GPU training or benchmark freeze.

## Scope and prerequisites

Use only the isolated checkout `/home/ubuntu/rootstep-benchmark` and the existing Isaac Lab container `isaac-arena-vscode`. Do not modify the active checkout `/home/ubuntu/sim2gate`, its untracked `deploy/brev/.env` or `mvp_runs/`. Record exact branch commit, Isaac Lab/Isaac Sim/RSL-RL/torch versions, GPU model, driver, and run timestamp. Gate 5 (165/165, reported by operator) validates the real DelayBuffer **with a minimal base motor stub**; it does not satisfy Gate 6.

**Approval boundary:** Launching a live Isaac Sim application can consume GPU resources. Obtain a separately approved bounded **nontraining simulation** budget and stop conditions before running the live portion. No PPO learner, optimizer, checkpoint generation, training loop or policy evaluation campaign is permitted.

## Required checks

1. **Config-to-instance identity:** Instantiate H0 and D2 Go2 task environments through the installed Isaac Lab runtime. Inspect the actual robot articulation's `base_legs` actuator instance, not only the config. Assert it is RootStep's `DelayedDCMotor` wrapping the actual Isaac Lab `DCMotor`, with actual Isaac Lab `DelayBuffer` objects. Capture class module/name and `class_type` identity.
2. **H0 vs D2 settings:** Verify H0 `min_delay=0,max_delay=6` and D2 `min_delay=0,max_delay=0`; verify identical non-intervention actuator fields and the intended training ramp at 48,000–76,800 physics steps. Verify the runtime play/evaluation mode explicitly and do not silently replace training settings.
3. **Reset sampling:** At controlled pre-ramp, mid-ramp and full-ramp counters, reset actuator and record sampled per-environment lag. Verify H0 bounds 0, 3, 6 respectively; D2 always zero; all three command channels share each sampled lag. If sampling is inaccessible through public API, document the instrumented read-only probe and validate it against observable command outputs.
4. **Live command propagation:** Run a small bounded number of physics steps under a deterministic safe command sequence. Observe target input, delayed targets, produced/applied motor efforts and articulation state; compare against predicted lag behavior and D2 zero-lag control. Do not infer command delay from robot trajectory alone.
5. **Partial reset:** Reset selected environment IDs and verify their lag/history are refreshed without changing nonreset environments. Preserve the actual observations, including any mismatch.
6. **No-training invariant:** Demonstrate no trainer/optimizer invocation, no checkpoint writes, no recipe mutation, and no unapproved GPU workload.

## Suggested execution bounds (not yet approved)

One H0 and one D2 instance, 1–2 environments each, only the steps necessary to exercise deterministic commands and reset behavior; no full training iterations. Predeclare a wall-clock ceiling and a maximum simulation-step count before launch. These are proposals, not permission to execute.

## Acceptance criteria

PASS only if all six checks are supported by saved machine-readable evidence and raw logs. FAIL for observed behavior mismatch. BLOCKED if runtime cannot initialize, resource approval is missing, or a required probe is unavailable. Do not downgrade missing evidence to PASS. In particular, real DelayBuffer+stub evidence cannot substitute for a live articulated robot test.

## Evidence to save

`GATE6_APPROVAL.md` (scope, approver, GPU/time caps); `GATE6_LIVE_ACTUATOR_REPORT.json` (status, checks, class/config identities, actual delays, target and effort traces, reset effects, environment versions, source commit); bounded console log; artifact hashes; and deviations. Do not store credentials or private env files.

## After Gate 6

Review pilot-only nominal qualification, independent manifestation criteria and stress grid; define the bounded runtime profiling procedure and seek its separate approval. Only then set GPU budget and request explicit authorization for the four-run pilot.
