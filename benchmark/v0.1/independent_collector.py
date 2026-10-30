#!/usr/bin/env python3
"""Tier-R trajectory collector: run one policy under one condition cell of the G2 manifest in the common P0 v5b
task, and write every episode's raw per-step trajectory for the independent truth code (independent_truth.py).

    /isaac-sim/python.sh benchmark/v0.1/independent_collector.py --manifest G2.json --policy s306 --cell F040 \\
        --out runs/s306__F040.jsonl.gz -- --task Sim2Gate-Velocity-Flat-Go2-P0-v5b --checkpoint <model_1999.pt> \\
        physics=isaacsim_physx env.sim.use_newton_actuators=false --external_callback sim2gate.v2.tasks.register

Environment setup reuses Sim2Gate's PLUMBING so the common task matches the historical nominal conditions exactly:
  - apply_profile: friction 0.8/0.6, startup mass/COM/payload randomization and pushes removed, task observation
    noise on, swept DC motor at zero delay;
  - command pinning: environment i holds command i mod n of the manifest's command set;
  - install: gymnasium.make / wrapper.step / runner.load hooks around Isaac Lab's play entry point.
It imports NO Sim2Gate scoring code or calibrated limits (robustness scoring, exploits_score, layer1_limits,
transfer, frozen threshold files). Truth is computed later, offline, from the raw trajectories.

Additionally, per cell:
  - static (dynamic = 0.75x) contact friction set on the task's physics_material event before the env is built;
  - every termination except the time limit is removed, so a fallen robot keeps being recorded and the
    independent fall detector, not the environment, decides falls;
  - action-side (primary, delay arm) or observation-side (secondary) latency of whole control steps
    (independent_truth.LatencyBuffer).

Runtime verification (revision 2), before the first recorded step; any failure aborts the run with no output:
  - the termination manager's active terms are exactly ['time_out'];
  - none of the profile's removed events is active;
  - every robot shape's static and dynamic friction, read back from the simulator, equals the cell's values;
  - every actuator is the swept DC motor with zero lag and full strength (latency comes only from the cell);
  - the EFFECTIVE robot-ground friction (robot readback combined with every ground material under PhysX's
    combine-mode rule, materials and modes read from the stage) equals the cell; robot and ground must not share
    a material; unbound robot shapes are allowed only if the ground has its own material, and when their runtime
    combine mode cannot be observed that limitation is recorded (contact_problems).
A diagnostic friction probe (|F_t| / F_n of sliding foot contacts) is recorded in the header; it never decides
validity. On failure a `<out>.diagnostic.json` with everything observed is written, and no run output.
Step synchronization (revision 2b), checked at every step for every environment (SyncTrace) and stored in the
header as `sync_checks`: capture-time episode length is consecutive from 1 after the collector's own reset; the
capture-time termination flag equals the done flag the step returns; a done environment's episode length is 0
after the step (the reset happened after the capture) and an open one's state is unchanged after the capture;
the captured command equals the pinned command; every environment ends at exactly max_episode_length. The first
call into step() resets every environment and discards the action the policy computed from the pre-reset
observation (it is never applied); the policy's next action is computed from the reset observation.
State is read inside reward computation, i.e. after physics and terminations but BEFORE the environment resets
ended episodes, so the terminal step of every episode is recorded (nothing is hidden by a reset). Any episode that
ends before the time limit is counted as unexpected and invalidates the run in the analysis.

Completion: the output is written to a temporary file, fsynced and renamed only after every episode has been
recorded, with header `completed: true` and a footer holding the record count and a digest of the records. The
process then exits at once with 0 (completed, file re-verified, step synchronization passed), 1 or 3, skipping
Isaac Sim's shutdown, which hung after a good run in S0 attempt 3.

Authorization: refuses to run unless the manifest is APPROVED and the requested stage is APPROVED and contains
this (policy, cell).

STATUS: not yet run. Running it (even as a smoke test) uses the GPU and needs explicit stage approval.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import independent_truth as it  # noqa: E402

FOOT_PATTERN = ".*_foot"
CONTACT_SENSOR = "contact_forces"
ROUND = 5


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cell(manifest, cell_id):
    cells = {c["id"]: c for c in manifest["cells"]}
    if cell_id not in cells:
        raise SystemExit(f"cell {cell_id!r} not in the manifest ({', '.join(sorted(cells))})")
    c = cells[cell_id]
    for k in ("static_friction", "latency_steps", "latency_side"):
        if k not in c:
            raise SystemExit(f"cell {cell_id}: missing {k}")
    if c["latency_side"] not in ("none", "observation", "action"):
        raise SystemExit(f"cell {cell_id}: latency_side must be none, observation or action")
    return c


def authorize(manifest, stage_id, policy, cell):
    """The run must belong to an APPROVED stage of an APPROVED manifest."""
    if manifest.get("status") != "APPROVED":
        raise SystemExit("the G2 manifest is not APPROVED; refusing to run")
    stages = {s["id"]: s for s in manifest.get("stages", [])}
    if stage_id not in stages:
        raise SystemExit(f"stage {stage_id!r} not in the manifest")
    st = stages[stage_id]
    if st.get("status") != "APPROVED":
        raise SystemExit(f"stage {stage_id} is not APPROVED; refusing to run")
    if [policy, cell] not in st["runs"]:
        raise SystemExit(f"({policy}, {cell}) is not a run of stage {stage_id}")
    return st


def runtime_problems(obs, cell, removed_events, tol=1e-4):
    """Compare what the simulator reports (plain numbers) with the cell. Returns a list of problems, empty if OK."""
    probs = []
    if obs["active_terminations"] != ["time_out"]:
        probs.append(f"active terminations {obs['active_terminations']}, expected ['time_out']")
    bad = sorted(set(obs["active_events"]) & set(removed_events))
    if bad:
        probs.append(f"events that should be removed are active: {bad}")
    mu = float(cell["static_friction"])
    if any(abs(x - mu) > tol for x in obs["friction_static"]):
        probs.append(f"static friction {obs['friction_static']} differs from the cell's {mu}")
    if any(abs(x - 0.75 * mu) > tol for x in obs["friction_dynamic"]):
        probs.append(f"dynamic friction {obs['friction_dynamic']} differs from {0.75 * mu}")
    if not obs["swept_motor"]:
        probs.append("not every actuator is the swept DC motor; was the nominal profile applied?")
    else:
        if obs["motor_lag_steps_max"] != 0.0:
            probs.append(f"motor lag {obs['motor_lag_steps_max']} physics steps, expected 0")
        if any(abs(x - 1.0) > 1e-6 for x in obs["motor_strength"]):
            probs.append(f"motor strength {obs['motor_strength']}, expected 1.0")
    lat, k, side = obs["latency"], int(cell["latency_steps"]), cell["latency_side"]
    want_obs, want_act = side == "observation" and k > 0, side == "action" and k > 0
    if lat["obs_buffer"] != want_obs or lat["action_buffer"] != want_act:
        probs.append(f"latency buffers (obs {lat['obs_buffer']}, action {lat['action_buffer']}) do not match "
                     f"the cell ({side}, {k} steps)")
    return probs


ENV0_ROOT = "/World/envs/env_0"


def resolve_env0_path(pattern, paths):
    """The env-0 prim matching an Isaac Lab prim-path pattern. Isaac Lab writes the environment part as a
    regex ({ENV_REGEX_NS}, env_.*, env_[^/]+, ...), so match the whole pattern as a regex against real prim paths
    and keep the shortest match under env_0."""
    import re
    rx = re.compile(pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_.*"))
    hits = sorted((p for p in paths if (p == ENV0_ROOT or p.startswith(ENV0_ROOT + "/")) and rx.fullmatch(p)),
                  key=len)
    if not hits:
        raise RuntimeError(f"no prim under {ENV0_ROOT} matches {pattern}")
    return hits[0]


COMBINE_PRIORITY = {"average": 0, "min": 1, "multiply": 2, "max": 3}   # PhysX: the higher-priority mode wins


def combine_friction(a, b, mode_a, mode_b):
    """Effective coefficient of a contact pair under PhysX's combine rule."""
    if mode_a not in COMBINE_PRIORITY or mode_b not in COMBINE_PRIORITY:
        raise ValueError(f"unknown combine mode {mode_a!r} / {mode_b!r}")
    mode = mode_a if COMBINE_PRIORITY[mode_a] >= COMBINE_PRIORITY[mode_b] else mode_b
    return {"average": (a + b) / 2.0, "min": min(a, b), "multiply": a * b, "max": max(a, b)}[mode]


def contact_problems(ev, cell, tol=1e-4):
    """Effective robot-ground friction (reviewer decision 1, 2026-10-30). ev: stage materials with combine modes,
    plus any runtime observation of the robot's combine mode. The robot's friction VALUES come from the simulator
    readback (checked equal to the cell elsewhere); this checks the effective coefficients of the contact pair.

    Robot collision shapes without a bound material are allowed if the ground has its own bound material, distinct
    from the scene default. Robot combine mode, in order of preference: observed at runtime; the bound material's
    mode; otherwise each mode the stack can produce for an unbound shape (the scene default's mode, and PhysX's
    default 'average' if the material was re-created), recorded as a limitation and refused if any material on the
    stage uses 'max'. Every resulting effective coefficient must equal the cell. Adds the derivation to ev."""
    probs, lim = [], ev.setdefault("limitations", [])
    mu, mu_d = float(cell["static_friction"]), 0.75 * float(cell["static_friction"])
    if ev.get("error"):
        return [f"contact material evidence unavailable: {ev['error']}"]
    ground, robot = ev.get("ground_materials") or [], ev.get("robot_shapes") or []
    unbound = int(ev.get("robot_unbound_shapes") or 0)
    default = (ev.get("scene_default_material") or {})
    if not ground:
        probs.append("no ground material found")
    if not ev.get("robot_collision_prims"):
        probs.append("no robot collision prims found")
    if ev.get("ground_unbound_collision_prims"):
        probs.append(f"{ev['ground_unbound_collision_prims']} ground collision prims use the scene default material")
    shared = sorted({r["material"] for r in robot} & {g["path"] for g in ground})
    if shared:
        probs.append(f"robot and ground share physics material(s) {shared}")
    if unbound and default.get("path") in {g["path"] for g in ground}:
        probs.append("unbound robot shapes and the ground both use the scene default material")
    modes, basis = {}, []
    for r in robot:
        modes.setdefault(r["mode"], "bound USD material")
    if unbound:
        obs = ev.get("robot_runtime_mode") or {}
        if obs.get("observable") and obs.get("modes"):
            for m_ in obs["modes"]:
                modes[m_] = "observed at runtime"
        else:
            cands = sorted({default.get("mode") or "average", "average"})
            for m_ in cands:
                modes.setdefault(m_, "inferred for unbound shapes")
            lim.append(f"the runtime combine mode of the {unbound} unbound robot shapes is not observable "
                       f"({obs.get('reason', 'no runtime query available')}); effective friction is computed for "
                       f"each mode the stack can produce for them: {cands}")
            maxed = [x["path"] for x in ev.get("stage_materials") or [] if x.get("mode") == "max"]
            if maxed:
                probs.append(f"material(s) {maxed} use 'max'; the unbound robot shapes' mode cannot be bounded")
            elif ev.get("stage_materials") is None:
                probs.append("stage material modes not recorded; the unbound robot shapes' mode cannot be bounded")
    ev["robot_modes_considered"] = modes
    eff_s, eff_d = set(), set()
    for g in ground:
        for m_ in modes:
            try:
                eff_s.add(round(combine_friction(mu, g["static"], m_, g["mode"]), 6))
                eff_d.add(round(combine_friction(mu_d, g["dynamic"], m_, g["mode"]), 6))
            except (ValueError, TypeError, KeyError) as e:
                probs.append(f"cannot combine robot mode {m_!r} with {g.get('path')}: {e}")
    ev["effective_static"], ev["effective_dynamic"] = sorted(eff_s), sorted(eff_d)
    if not eff_s:
        probs.append("no effective coefficient could be computed")
    if any(abs(x - mu) > tol for x in eff_s):
        probs.append(f"effective robot-ground static friction {sorted(eff_s)} differs from the cell's {mu}")
    if any(abs(x - mu_d) > tol for x in eff_d):
        probs.append(f"effective robot-ground dynamic friction {sorted(eff_d)} differs from {mu_d}")
    return probs


class FrictionProbe:
    """Diagnostic only (reviewer decision 2): |F_t| / F_n of foot contacts while the foot slides. Never a pass/fail
    criterion, and a close match to the dynamic coefficient is not proof of it."""
    LIMITATIONS = [
        "net contact force of the foot body: several contact points and any non-ground contact are summed",
        "slip is detected from the foot body's horizontal velocity, which includes rolling of the spherical foot",
        "transient and impact samples are included; the ratio equals the coefficient only in steady sliding",
        "few sliding samples are expected on high-friction cells",
    ]

    def __init__(self, min_normal_n=10.0, min_slip_mps=0.05, max_keep=200000):
        self.fn_min, self.v_min, self.max_keep = float(min_normal_n), float(min_slip_mps), int(max_keep)
        self.contact = self.sliding = 0
        self.ratios, self.error = [], None

    def add(self, forces_xyz, vel_xy, open_):
        """forces_xyz: [env][foot][3] world contact force; vel_xy: [env][foot][2] foot velocity; open_: [env]."""
        for i, ok in enumerate(open_):
            if not ok:
                continue
            for f, v in zip(forces_xyz[i], vel_xy[i]):
                fn = float(f[2])
                if fn <= self.fn_min:
                    continue
                self.contact += 1
                if math.hypot(float(v[0]), float(v[1])) <= self.v_min:
                    continue
                self.sliding += 1
                if len(self.ratios) < self.max_keep:
                    self.ratios.append(math.hypot(float(f[0]), float(f[1])) / fn)

    def result(self, mu_dynamic):
        q = it._quantile
        r = sorted(self.ratios)
        return {"role": "diagnostic only; not a pass/fail criterion", "error": self.error,
                "thresholds": {"min_normal_n": self.fn_min, "min_slip_mps": self.v_min},
                "contact_samples": self.contact, "sliding_samples": self.sliding, "ratios_kept": len(r),
                "median": q(r, 0.5), "q25": q(r, 0.25), "q75": q(r, 0.75),
                "cell_dynamic_coefficient": mu_dynamic, "limitations": list(self.LIMITATIONS)}


class SyncTrace:
    """Per-step evidence, for every open environment, that the recorded samples are consecutive post-physics
    states, that the last one is the actual terminal transition, and that no reset state is recorded.
    Pure Python (lists in, counts out), so its logic is unit-tested without Isaac Lab.

    At capture (inside reward computation) the environment reports its episode length and termination flag;
    after the step returns, its episode length again and the base position. For a sample taken at policy step k:
      - capture episode length must be k (consecutive, starting at 1 after the collector's reset);
      - the capture-time termination flag must equal the done flag the step returned;
      - done: the post-step episode length must be 0 (the reset happened AFTER the capture);
      - not done: the post-step episode length and base position must equal the captured ones.
    The command at capture must equal the pinned command. Every environment must end exactly at max_steps."""

    def __init__(self, expected_cmds, max_steps, tol=1e-6):
        self.cmds, self.max_steps, self.tol = [list(map(float, c)) for c in expected_cmds], int(max_steps), tol
        self.n = len(self.cmds)
        self.c = {"steps": 0, "samples_checked": 0, "capture_length_not_consecutive": 0,
                  "termination_flag_mismatch": 0, "done_without_reset_after_capture": 0,
                  "state_changed_after_capture": 0, "command_mismatch": 0, "reset_moved_robot": 0,
                  "stale_env_samples": 0, "steps_without_physics_advance": 0}
        self._prev_pos = None
        self.max_cmd_dev = self.max_state_dev = 0.0
        self.reset_length_max, self.terminal = None, [None] * self.n

    def on_reset(self, ep_len):
        self.reset_length_max = max(int(x) for x in ep_len)

    def on_step(self, k, open_, cap_len, cap_flag, cap_pos, cap_cmd, dones, post_len, post_pos):
        c = self.c
        c["steps"] += 1
        moved_any, checked = False, 0
        for i in range(self.n):
            if not open_[i]:
                continue
            c["samples_checked"] += 1
            checked += 1
            if self._prev_pos is not None and self._prev_pos[i] is not None:
                if list(map(float, cap_pos[i])) == self._prev_pos[i]:
                    c["stale_env_samples"] += 1                      # identical to the previous capture
                else:
                    moved_any = True
            else:
                moved_any = True
            if int(cap_len[i]) != k:
                c["capture_length_not_consecutive"] += 1
            if bool(cap_flag[i]) != bool(dones[i]):
                c["termination_flag_mismatch"] += 1
            dc = max(abs(float(a) - b) for a, b in zip(cap_cmd[i], self.cmds[i]))
            self.max_cmd_dev = max(self.max_cmd_dev, dc)
            if dc > 1e-5:
                c["command_mismatch"] += 1
            dp = max(abs(float(a) - float(b)) for a, b in zip(cap_pos[i], post_pos[i]))
            if dones[i]:
                if int(post_len[i]) != 0:
                    c["done_without_reset_after_capture"] += 1
                if dp > self.tol:
                    c["reset_moved_robot"] += 1                       # the reset is visible, and not recorded
                self.terminal[i] = {"step": k, "capture_length": int(cap_len[i]), "flag": bool(cap_flag[i])}
            else:
                self.max_state_dev = max(self.max_state_dev, dp)
                if int(post_len[i]) != int(cap_len[i]) or dp > self.tol:
                    c["state_changed_after_capture"] += 1
        if checked and not moved_any:
            c["steps_without_physics_advance"] += 1                  # no environment changed: not post-physics
        self._prev_pos = [list(map(float, cap_pos[i])) if open_[i] else None for i in range(self.n)]

    def result(self, lengths):
        c, probs = self.c, []
        if self.reset_length_max != 0:
            probs.append(f"episode length after the collector's reset is {self.reset_length_max}, expected 0")
        for k in ("capture_length_not_consecutive", "termination_flag_mismatch", "done_without_reset_after_capture",
                  "state_changed_after_capture", "command_mismatch", "steps_without_physics_advance"):
            if c[k]:
                probs.append(f"{k}: {c[k]}")
        bad_end = sum(1 for t in self.terminal if t is None or t["step"] != self.max_steps
                      or t["capture_length"] != self.max_steps or not t["flag"])
        if bad_end:
            probs.append(f"{bad_end} environments did not end with a terminal transition at step {self.max_steps}")
        if any(n != self.max_steps for n in lengths):
            probs.append(f"recorded lengths {min(lengths)}-{max(lengths)}, expected exactly {self.max_steps}")
        return dict(c, max_command_deviation=self.max_cmd_dev, max_state_change_after_capture=self.max_state_dev,
                    reset_episode_length_max=self.reset_length_max, terminal_steps=sorted(
                        {t["step"] for t in self.terminal if t}), problems=probs, ok=not probs)


class EpisodeLog:
    """Accumulates per-environment samples (plain Python lists) and emits independent-truth episode records.
    Pure Python, so its behaviour is unit-tested without Isaac Lab."""

    def __init__(self, n_envs, foot_names, dt):
        self.n, self.feet, self.dt = n_envs, list(foot_names), float(dt)
        self.samples = [[] for _ in range(n_envs)]
        self.open = [True] * n_envs

    def add(self, snap, done=None):
        """snap: dict of per-env lists for one control step, read BEFORE resets; done: per-env bool, True if this
        step ended the episode. The terminal sample is recorded, then the environment's log is closed."""
        for i in range(self.n):
            if not self.open[i]:
                continue
            self.samples[i].append({
                "h": snap["base_height_m"][i], "g": snap["projected_gravity_b"][i],
                "v": snap["base_lin_vel_b_xy"][i], "c": snap["cmd_vel_xy"][i], "p": snap["base_pos_w_xy"][i],
                "f": snap["foot_force_n"][i], "q": snap["foot_pos_w_xy"][i]})
            if done is not None and done[i]:
                self.open[i] = False

    @property
    def all_closed(self):
        return not any(self.open)

    def record(self, i):
        s = self.samples[i]
        r = lambda x: round(float(x), ROUND)  # noqa: E731
        return {"schema": it.SCHEMA_VERSION, "env": i, "dt": self.dt,
                "base_height_m": [r(x["h"]) for x in s],
                "projected_gravity_b": [[r(v) for v in x["g"]] for x in s],
                "base_lin_vel_b_xy": [[r(v) for v in x["v"]] for x in s],
                "cmd_vel_xy": [[r(v) for v in x["c"]] for x in s],
                "base_pos_w_xy": [[r(v) for v in x["p"]] for x in s],
                "feet": {name: {"contact_force_n": [r(x["f"][j]) for x in s],
                                "pos_w_xy": [[r(v) for v in x["q"][j]] for x in s]}
                         for j, name in enumerate(self.feet)}}

    def lengths(self):
        return [len(x) for x in self.samples]

    def write(self, path, header):
        """Write header, records and footer to a temporary file, then rename: a file at `path` is always complete."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = str(path) + ".tmp"
        h = hashlib.sha256()
        with gzip.open(tmp, "wt") as f:
            # default=str: a non-JSON value in the evidence (e.g. a USD token) must not lose a completed run
            f.write(json.dumps({"header": dict(header, completed=True)}, default=str) + "\n")
            for i in range(self.n):
                rec = self.record(i)
                it.validate_episode(rec)
                line = json.dumps(rec, separators=(",", ":"))
                h.update(line.encode())
                f.write(line + "\n")
            f.write(json.dumps({"footer": {"records": self.n, "records_sha256": h.hexdigest()}}) + "\n")
        fd = os.open(tmp, os.O_RDONLY)                               # durable before rename (and before a hard exit)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        dfd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def _hard_exit(code):
    """End the process now, with `code`. S0 attempt 3: after the output was written and verified, Isaac Sim's
    SimulationApp.close() (run by the play script's launch context when the play loop is interrupted) hung until a
    2-minute watchdog killed it, so a good run exited non-zero. The output is closed, fsynced and renamed before
    this is called, and nothing after the play loop is needed, so Kit's shutdown is skipped."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def verify_written(path):
    """Re-read a run file: header marked completed, footer present and matching the records. Returns problems."""
    try:
        with gzip.open(path, "rt") as f:
            lines = [json.loads(x) for x in f if x.strip()]
    except (OSError, EOFError, ValueError) as e:
        return [f"unreadable: {type(e).__name__}"]
    if not lines or (lines[0].get("header") or {}).get("completed") is not True:
        return ["header missing or not marked completed"]
    footer = lines[-1].get("footer") if len(lines) > 1 else None
    if footer is None:
        return ["no footer"]
    recs = lines[1:-1]
    h = hashlib.sha256()
    for r in recs:
        h.update(json.dumps(r, separators=(",", ":")).encode())
    if footer.get("records") != len(recs) or footer.get("records_sha256") != h.hexdigest():
        return ["footer does not match the records"]
    return []


class Collector:
    """Hooks for sim2gate.checks.isaaclab_conditions.install: on_env_cfg, step, record_checkpoint."""

    def __init__(self, manifest, manifest_path, policy_id, cell, out_path, play_args, stage_id=None):
        self.manifest, self.policy_id, self.cell, self.out = manifest, policy_id, cell, out_path
        self.stage_id, self.completed, self.runtime_checks = stage_id, False, {}
        self.manifest_sha256 = _sha_file(manifest_path)
        self.play_args = list(play_args)
        cs = manifest["command_set"]
        from sim2gate.checks import commands as command_sets
        self.command_sets = command_sets
        self.command_doc = command_sets.load_command_set(str(HERE / cs["path"]))
        if self.command_doc["sha256"] != cs["sha256"]:
            raise SystemExit("command set sha256 differs from the manifest")
        self.n_envs = len(self.command_doc["commands"]) * int(manifest["episodes_per_cell_repeat"])
        self.checkpoints, self.cfg_changes, self.started = [], {}, False
        self.obs_buf = self.act_buf = None
        self._pending, self._pending_sync, self._pending_probe, self._in_step, self.early_dones = None, None, None, False, 0
        self.probe = FrictionProbe()
        self._exit = _hard_exit
        self.sync_checks = None

    def record_checkpoint(self, path):
        try:
            self.checkpoints.append({"path": os.path.abspath(path), "sha256": _sha_file(path)})
        except OSError as e:
            self.checkpoints.append({"path": str(path), "sha256": None, "error": str(e)})

    # -- before the env is built --
    def on_env_cfg(self, cfg):
        from sim2gate.checks.isaaclab_conditions import apply_profile
        self.cfg_changes["profile_record"] = apply_profile(cfg)
        cfg.scene.num_envs = self.n_envs
        mu = float(self.cell["static_friction"])
        pm = cfg.events.physics_material.params
        pm["static_friction_range"] = (mu, mu)
        pm["dynamic_friction_range"] = (0.75 * mu, 0.75 * mu)
        self.cfg_changes["contact_material"] = {"static": mu, "dynamic": 0.75 * mu}
        removed = []
        terms = cfg.terminations
        for name in list(vars(terms)):
            if name != "time_out" and getattr(terms, name, None) is not None and not name.startswith("_"):
                setattr(terms, name, None)
                removed.append(name)
        if getattr(terms, "time_out", None) is None:
            raise RuntimeError("the task has no time_out termination; episodes would never end")
        self.cfg_changes["removed_terminations"] = removed
        mats = {}
        for name, mat in (("terrain", getattr(getattr(cfg.scene, "terrain", None), "physics_material", None)),
                          ("sim_default", getattr(cfg.sim, "physics_material", None))):
            if mat is not None:
                mats[name] = {k: getattr(mat, k, None) for k in ("static_friction", "dynamic_friction",
                                                                 "friction_combine_mode", "restitution")}
        self.cfg_changes["configured_ground_materials"] = mats

    # -- every policy step --
    def step(self, wrapper, orig_step, actions):
        env = wrapper.unwrapped
        if not self.started:
            import torch
            self._setup(env)
            self.started = True
            obs, extras = wrapper.reset()
            self.sync.on_reset(env.episode_length_buf.tolist())
            obs = self._delay_obs(obs)
            n = wrapper.num_envs
            return obs, torch.zeros(n, device=env.device), torch.zeros(n, dtype=torch.long, device=env.device), \
                dict(extras or {}, time_outs=torch.zeros(n, dtype=torch.bool, device=env.device))
        if self.act_buf is not None:
            actions = self.act_buf(actions.clone())
        self._pending, self._in_step = None, True
        try:
            out = orig_step(wrapper, actions)
        finally:
            self._in_step = False
        if self._pending is None:
            raise RuntimeError("reward computation did not run inside the step; no pre-reset state was captured")
        dones = out[2].bool().tolist()
        self.steps += 1
        if any(dones) and self.steps < self.max_steps:
            self.early_dones += sum(dones)
        t = lambda x: x.torch if hasattr(x, "torch") else x  # noqa: E731
        self.sync.on_step(self.steps, list(self.log.open), self._pending_sync["len"], self._pending_sync["flag"],
                          self._pending_sync["pos"], self._pending_sync["cmd"], dones,
                          env.episode_length_buf.tolist(), t(env.scene["robot"].data.root_pos_w).tolist())
        if self._pending_probe is not None:
            self.probe.add(self._pending_probe[0], self._pending_probe[1], list(self.log.open))
        self.log.add(self._pending, done=dones)
        if self.log.all_closed or self.steps >= self.max_steps + 5:
            self._finish(env)
            self._exit(self.exit_code())                           # does not return in a real run
        return (self._delay_obs(out[0]),) + tuple(out[1:])

    def _delay_obs(self, obs):
        return self.obs_buf(obs.clone()) if self.obs_buf is not None else obs

    def _setup(self, env):
        if env.num_envs != self.n_envs:
            raise RuntimeError(f"{env.num_envs} environments, expected {self.n_envs}")
        self._undo_cmd, self.cmd_info = self.command_sets.pin(env, self.command_doc, tile=True)
        robot, sensor = env.scene["robot"], env.scene.sensors[CONTACT_SENSOR]
        self.foot_ids_robot, self.foot_names = robot.find_bodies(FOOT_PATTERN)
        self.foot_ids_sensor, sensor_names = sensor.find_bodies(FOOT_PATTERN)
        if list(sensor_names) != list(self.foot_names) or len(self.foot_names) != 4:
            raise RuntimeError(f"foot bodies differ: robot {self.foot_names}, sensor {sensor_names}")
        k, side = int(self.cell["latency_steps"]), self.cell["latency_side"]
        if side == "observation" and k > 0:
            self.obs_buf = it.LatencyBuffer(k)
        elif side == "action" and k > 0:
            self.act_buf = it.LatencyBuffer(k)
        self.log = EpisodeLog(env.num_envs, self.foot_names, float(env.step_dt))
        self.steps, self.max_steps = 0, int(env.max_episode_length)
        m = len(self.command_doc["commands"])
        self.sync = SyncTrace([self.command_doc["commands"][i % m] for i in range(env.num_envs)], self.max_steps)
        self._verify_runtime(env)
        self._install_prereset_hook(env)

    def _install_prereset_hook(self, env):
        """Capture state inside reward computation: after physics and terminations, before ended episodes reset."""
        mgr = env.reward_manager
        orig_compute = mgr.compute

        def compute(*a, **k):
            out = orig_compute(*a, **k)
            if self._in_step:
                self._pending = self._snapshot(env)
                self._pending_sync = self._sync_snapshot(env)
                self._pending_probe = self._probe_snapshot(env)
            return out
        mgr.compute = compute

    def _probe_snapshot(self, env):
        """Full foot contact force and foot horizontal velocity, for the diagnostic friction probe. Never fatal."""
        if self.probe.error:
            return None
        try:
            t = lambda x: x.torch if hasattr(x, "torch") else x  # noqa: E731
            f = t(env.scene.sensors[CONTACT_SENSOR].data.net_forces_w)[:, self.foot_ids_sensor, :]
            v = t(env.scene["robot"].data.body_lin_vel_w)[:, self.foot_ids_robot, :2]
            return f.tolist(), v.tolist()
        except Exception as e:
            self.probe.error = f"{type(e).__name__}: {e}"
            return None

    def _sync_snapshot(self, env):
        t = lambda x: x.torch if hasattr(x, "torch") else x  # noqa: E731
        flag = getattr(env, "reset_buf", None)
        if flag is None:
            flag = env.termination_manager.dones
        return {"len": env.episode_length_buf.tolist(), "flag": flag.bool().tolist(),
                "pos": t(env.scene["robot"].data.root_pos_w).tolist(),
                "cmd": env.command_manager.get_term("base_velocity").vel_command_b[:, :3].tolist()}

    def _contact_evidence(self, env):
        """Physics materials as authored on the stage: every ground collision prim's bound material (or the scene
        default material if unbound) and the robot's (env 0) collision prims' bound materials, with combine modes."""
        try:
            from pxr import UsdPhysics, UsdShade
            stage = env.scene.stage

            def mat_info(prim):
                api = UsdPhysics.MaterialAPI(prim)
                mode = prim.GetAttribute("physxMaterial:frictionCombineMode")
                return {"path": str(prim.GetPath()), "static": api.GetStaticFrictionAttr().Get(),
                        "dynamic": api.GetDynamicFrictionAttr().Get(),
                        "mode": (mode.Get() if mode and mode.HasAuthoredValue() else "average")}

            def bound(prim):
                m, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                return m.GetPrim() if m else None

            from pxr import Usd

            def collision_prims(root):
                r = stage.GetPrimAtPath(root)
                if not r or not r.IsValid():
                    raise RuntimeError(f"prim {root} not found")
                # instanceable assets keep their collision prims inside instance proxies
                return [q for q in Usd.PrimRange(r, Usd.TraverseInstanceProxies())
                        if q.HasAPI(UsdPhysics.CollisionAPI)]

            default = None
            for q in stage.Traverse():
                if q.IsA(UsdPhysics.Scene):
                    default = bound(q)
                    break
            ground, gpaths, g_unbound = [], set(), 0
            for q in collision_prims(env.scene.terrain.cfg.prim_path):
                mp = bound(q) or default
                g_unbound += bound(q) is None
                if mp is not None and str(mp.GetPath()) not in gpaths:
                    gpaths.add(str(mp.GetPath()))
                    ground.append(mat_info(mp))
            pattern = env.scene["robot"].cfg.prim_path
            env0 = stage.GetPrimAtPath(ENV0_ROOT)
            search = Usd.PrimRange(env0) if env0 and env0.IsValid() else stage.Traverse()
            rpath = resolve_env0_path(pattern, (str(q.GetPath()) for q in search))
            robot, r_unbound, rmats = [], 0, {}
            for q in collision_prims(rpath):
                mp = bound(q)
                if mp is None:
                    r_unbound += 1
                    continue
                info = rmats.setdefault(str(mp.GetPath()), mat_info(mp))
                robot.append({"prim": str(q.GetPath()), "material": info["path"], "mode": info["mode"]})
            stage_mats = [mat_info(q) for q in stage.Traverse() if q.HasAPI(UsdPhysics.MaterialAPI)]
            runtime_mode = {"observable": False}
            try:
                view = env.scene["robot"].root_view
                names = sorted(n for n in dir(view) if "combine" in n.lower())
                runtime_mode["view_methods"] = names
                if not names:
                    runtime_mode["reason"] = "the simulator's tensor view exposes no combine-mode query"
            except Exception as e:
                runtime_mode["reason"] = f"{type(e).__name__}: {e}"
            return {"robot_prim_pattern": pattern, "ground_materials": ground, "stage_materials": stage_mats,
                    "robot_runtime_mode": runtime_mode,
                    "ground_collision_prims": len(collision_prims(env.scene.terrain.cfg.prim_path)),
                    "ground_unbound_collision_prims": g_unbound,
                    "scene_default_material": mat_info(default) if default is not None else None,
                    "robot_prim": rpath, "robot_collision_prims": len(robot) + r_unbound,
                    "robot_shapes": robot, "robot_unbound_shapes": r_unbound,
                    "robot_materials_usd": list(rmats.values())}
        except Exception as e:                                       # evidence, not a crash: reported as a problem
            return {"error": f"{type(e).__name__}: {e}"}

    def _write_diagnostic(self):
        """On a failed runtime check: write what was observed next to the (never-created) output, then abort."""
        path = str(self.out) + ".diagnostic.json"
        with open(path, "w") as f:
            json.dump({"policy_id": self.policy_id, "cell": self.cell, "stage": self.stage_id,
                       "runtime_checks": self.runtime_checks, "cfg_changes": self.cfg_changes,
                       "note": "runtime verification failed; no run output written"}, f, indent=1, default=str)
        return path

    def _verify_runtime(self, env):
        """Effective configuration as the simulator holds it, not as configured. Raises on any mismatch."""
        import torch
        import warp as wp
        from sim2gate.checks.isaaclab_conditions import REMOVED_EVENTS
        t = lambda x: x.torch if hasattr(x, "torch") else x  # noqa: E731
        active = list(getattr(env.termination_manager, "active_terms", []))
        ev = getattr(env.event_manager, "active_terms", {})
        events = sorted({n for terms in (ev.values() if isinstance(ev, dict) else [ev]) for n in terms})
        robot = env.scene["robot"]
        mat = wp.to_torch(robot.root_view.get_material_properties()).float()
        st, dy = mat[..., 0], mat[..., 1]
        motors = list(robot.actuators.values())
        swept = bool(motors) and all(hasattr(m, "sim2gate_lag") for m in motors)
        lag = torch.cat([t(m.sim2gate_lag).flatten().float() for m in motors]) if swept else None
        strength = torch.cat([t(m.sim2gate_strength).flatten().float() for m in motors]) if swept else None
        obs = {"active_terminations": active, "active_events": events,
               "friction_static": [float(st.min()), float(st.max())],
               "friction_dynamic": [float(dy.min()), float(dy.max())], "friction_shapes": list(st.shape),
               "swept_motor": swept,
               "motor_lag_steps_max": float(lag.abs().max()) if swept else None,
               "motor_strength": [float(strength.min()), float(strength.max())] if swept else None,
               "latency": {"side": self.cell["latency_side"], "control_steps": int(self.cell["latency_steps"]),
                           "ms": int(self.cell["latency_steps"]) * float(env.step_dt) * 1000.0,
                           "obs_buffer": self.obs_buf is not None, "action_buffer": self.act_buf is not None}}
        contact = self._contact_evidence(env)
        problems = runtime_problems(obs, self.cell, REMOVED_EVENTS) + contact_problems(contact, self.cell)
        self.runtime_checks = dict(obs, contact=contact, problems=problems, ok=not problems)
        if problems:
            where = self._write_diagnostic()
            raise RuntimeError("runtime verification failed (" + where + "): " + "; ".join(problems))

    def _snapshot(self, env):
        t = lambda x: x.torch if hasattr(x, "torch") else x  # noqa: E731  (warp arrays in Isaac Lab 3.x)
        d = env.scene["robot"].data
        pos = t(d.root_pos_w)
        feet_pos = t(d.body_pos_w)[:, self.foot_ids_robot, :2]
        forces = t(env.scene.sensors[CONTACT_SENSOR].data.net_forces_w)[:, self.foot_ids_sensor, 2]
        cmd = env.command_manager.get_term("base_velocity").vel_command_b[:, :2]
        return {"base_height_m": pos[:, 2].tolist(), "base_pos_w_xy": pos[:, :2].tolist(),
                "projected_gravity_b": t(d.projected_gravity_b).tolist(),
                "base_lin_vel_b_xy": t(d.root_lin_vel_b)[:, :2].tolist(), "cmd_vel_xy": cmd.tolist(),
                "foot_force_n": forces.tolist(), "foot_pos_w_xy": feet_pos.tolist()}

    def exit_code(self):
        """0 only for a completed run whose file re-verifies and whose step synchronization passed."""
        if not self.completed:
            print(f"[independent] {self.policy_id}: NOT completed", file=sys.stderr)
            return 1
        problems = verify_written(self.out)
        if problems:
            print(f"[independent] {self.policy_id}: output fails verification: {problems}", file=sys.stderr)
            return 1
        if not (self.sync_checks or {}).get("ok"):
            print(f"[independent] {self.policy_id}: step synchronization failed: {(self.sync_checks or {}).get('problems')}",
                  file=sys.stderr)
            return 3
        print(f"[independent] {self.policy_id} {self.cell['id']}: completed, verified, exit 0")
        return 0

    def _finish(self, env):
        ck = self.checkpoints[-1] if self.checkpoints else None
        expected = self.manifest["policies"][self.policy_id]["checkpoint_sha256"]
        lengths = self.log.lengths()
        self.sync_checks = self.sync.result(lengths)
        import importlib
        deps = {}
        for mod in ("sim2gate.checks.isaaclab_conditions", "sim2gate.checks.commands"):
            try:
                f = importlib.import_module(mod).__file__
                deps[mod] = {"path": f, "sha256": _sha_file(f)}
            except Exception as e:                                   # recorded, not fatal: provenance only
                deps[mod] = {"error": f"{type(e).__name__}: {e}"}
        header = {"sync_checks": self.sync_checks, "dependency_files": deps,
                  "friction_probe": self.probe.result(0.75 * float(self.cell["static_friction"])),"tool": "independent_collector", "tool_sha256": _sha_file(__file__), "stage": self.stage_id,
                  "runtime_checks": self.runtime_checks, "unexpected_early_dones": self.early_dones,
                  "episode_lengths": {"min": min(lengths), "max": max(lengths)},
                  "truth_lib_sha256": _sha_file(HERE / "independent_truth.py"),
                  "manifest_sha256": self.manifest_sha256, "policy_id": self.policy_id, "cell": self.cell,
                  "checkpoint": ck, "checkpoint_matches_manifest": bool(ck and ck.get("sha256") == expected),
                  "task": next((self.play_args[i + 1] for i, a in enumerate(self.play_args[:-1]) if a == "--task"), None),
                  "play_args": self.play_args, "cfg_changes": self.cfg_changes, "commands": self.cmd_info,
                  "command_set_sha256": self.command_doc["sha256"], "num_envs": env.num_envs,
                  "step_dt": float(env.step_dt), "episode_steps": self.max_steps, "steps_run": self.steps,
                  "foot_names": list(self.foot_names)}
        self.log.write(self.out, header)
        self.completed = True
        print(f"[independent] {self.policy_id} {self.cell['id']}: {env.num_envs} episodes, {self.steps} steps "
              f"-> {self.out}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, rest = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", required=True, help="an APPROVED stage of the manifest that contains this run")
    a = p.parse_args(ours)
    with open(a.manifest) as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "APPROVED":
        raise SystemExit("the G2 manifest is not APPROVED; refusing to run")
    if a.policy not in manifest["policies"]:
        raise SystemExit(f"policy {a.policy!r} not in the manifest")
    stage = authorize(manifest, a.stage, a.policy, a.cell)
    task = next((rest[i + 1] for i, x in enumerate(rest[:-1]) if x == "--task"), None)
    if task != manifest["common_task"]:
        raise SystemExit(f"--task must be the common task {manifest['common_task']!r}, got {task!r}")
    missing = [x for x in manifest.get("play_args_fixed", []) if x not in rest]
    if missing:
        raise SystemExit(f"fixed play arguments missing from the command line: {missing}")
    if any(x == "--num_envs" or x.startswith("--num_envs=") for x in rest):
        raise SystemExit("--num_envs is set from the manifest; do not pass it")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; runs are written once")
    col = Collector(manifest, a.manifest, a.policy, load_cell(manifest, a.cell), a.out, rest, stage_id=stage["id"])
    from sim2gate.checks.isaaclab_conditions import install
    install(col)
    import runpy
    sys.argv = ["play_rsl_rl"] + rest + ["--num_envs", str(col.n_envs)]
    try:
        runpy.run_module("isaaclab_rl.entrypoints.backends.play_rsl_rl", run_name="__main__")
    except KeyboardInterrupt:
        pass
    # normally not reached: a finished run ends the process from step() via _hard_exit
    if not col.completed and os.path.exists(a.out + ".tmp"):
        os.remove(a.out + ".tmp")
    return col.exit_code()


if __name__ == "__main__":
    sys.exit(main())
