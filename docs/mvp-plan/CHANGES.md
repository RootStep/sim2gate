# Changes to the frozen MVP plan

The frozen plan (`MVP_PLAN_FROZEN_2026-10-12.md`, SHA-256 `9b9d2c85…dc805677`) is never edited. Each change is
logged here with its date, reason, what it affects, and who approved it. Acceptance criteria A1–A6 are unchanged
unless an entry says otherwise.

## C1 — Who approves plan changes (Oct 13, 2026)

- **Was:** the plan names Vihaan as the approver ("It needs Vihaan's approval before Phase 1 starts; after approval
  it is frozen and any change is logged").
- **Now:** VJ approves changes to the frozen plan from here on. Vihaan's role as lead researcher, and his role as
  holder of the blinding key, are unchanged.
- **Approved by:** VJ, Oct 13, 2026.

## C2 — Checks run on Isaac Lab directly; the Arena port follows validation (Oct 13, 2026)

- **Was:** checks 1, 2 and 5 are built as Isaac Lab-Arena plug-ins on a Go2 embodiment and velocity-tracking task
  ported into Arena. Check 1 uses Arena variation plug-ins and its experiment runner, and check 5 uses Arena's engine
  switch.
- **Now:** for the MVP, all three checks run on Isaac Lab 3.0 directly, on Isaac Lab's own Go2 flat
  velocity-tracking task and its RSL-RL train and play entry points.
  - Check 1 applies its physics variations through Isaac Lab's event terms and configuration overrides.
  - Check 5 uses Isaac Lab's physics selector (`physics=physx` vs `physics=newton_mjwarp`).
  - The Arena port (Go2 embodiment, task, metric and variation plug-ins), and contributing the physics variations
    and exploit metrics back to Arena, come after the MVP validation.
- **Why:**
  - Arena at the pinned commit (2cc9e97) has no Go2 embodiment and no locomotion task, only manipulation tasks.
    Porting them means rebuilding the task with observations that match the trained policies exactly, which is
    several days of work. It also adds a new way to fail parity criterion A5 that has nothing to do with the
    checks themselves.
  - Isaac Lab 3.0 now provides what the plan wanted from Arena for these checks: a PhysX/Newton (MuJoCo-Warp)
    switch on the same task, and event-based physics randomization.
  - The policies, observations, actions and task code stay identical between check runs. Only the measured
    condition changes, which is what check 5's parity conditions require.
- **Affects:**
  - The "Built on Isaac Lab-Arena" section, the sentences in checks 1, 2 and 5 that name Arena plug-ins, and the
    "Decisions for Vihaan" item on building the checks as Arena plug-ins.
  - Check 1's joint sweep (report only) uses its own uniform sampler and analysis instead of Arena's sensitivity
    analysis, with the same uniform-draw requirement.
  - Thresholds, flag rules, the test set, blinding and acceptance criteria are unchanged.
- **Approved by:** VJ, Oct 13, 2026.
