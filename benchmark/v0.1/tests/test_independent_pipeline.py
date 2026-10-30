"""Synthetic tests for the tier-R collector log, collector guards and offline analysis (no Isaac Lab, no torch)."""
import ast
import contextlib
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import independent_analysis as ia
import independent_collector as ic
import independent_truth as it

MANIFEST = json.loads((HERE / "G2_TIER_R_MANIFEST.json").read_text())
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def snap(n, t, height=0.32, vx=0.5, slide=0.0):
    """One control step for n envs: trotting feet (stance 12 of 20 steps), base moving forward at vx."""
    on = (t % 20) < 12
    fx = 0.1 * (t // 20) + (slide * 0.02 * (t % 20) if on else 0.0)
    return {"base_height_m": [height] * n, "projected_gravity_b": [[0.0, 0.0, -1.0]] * n,
            "base_lin_vel_b_xy": [[vx, 0.0]] * n, "cmd_vel_xy": [[0.5, 0.0]] * n,
            "base_pos_w_xy": [[0.5 * 0.02 * t, 0.0]] * n,
            "foot_force_n": [[60.0 if on else 0.0] * 4] * n, "foot_pos_w_xy": [[[fx, 0.0]] * 4] * n}


def small_manifest(n=20, cells=("N080", "F040", "F045", "A1", "A2")):
    """The real manifest with fewer cells and n episodes per run, so synthetic runs stay small."""
    m = copy.deepcopy(MANIFEST)
    m["cells"] = [c for c in m["cells"] if c["id"] in cells]
    m["episodes_per_cell_repeat"] = 1
    m["command_set"]["n_commands"] = n
    return m


def good_header(m, sha, policy, cell, n, T):
    return {"manifest_sha256": sha, "policy_id": policy, "cell": next(c for c in m["cells"] if c["id"] == cell),
            "checkpoint": {"sha256": m["policies"][policy]["checkpoint_sha256"]}, "checkpoint_matches_manifest": True,
            "command_set_sha256": m["command_set"]["sha256"], "task": m["common_task"], "num_envs": n,
            "runtime_checks": {"ok": True, "problems": []}, "sync_checks": {"ok": True, "problems": []},
            "unexpected_early_dones": 0, "episode_steps": T,
            "steps_run": T, "play_args": ["--task", m["common_task"], "--checkpoint", "x.pt"] + m["play_args_fixed"]}


def write_run(path, m, sha, policy, cell, n, fall_envs=0, T=200, slide=0.0, header_changes=None, short_env=None):
    log = ic.EpisodeLog(n, FEET, 0.02)
    for t in range(T):
        s = snap(n, t, slide=slide)
        if t >= 100:
            s["base_height_m"] = [0.05 if i < fall_envs else 0.32 for i in range(n)]
        log.add(s, done=[t == T - 1 or (i == short_env and t == T // 2) for i in range(n)])
    header = dict(good_header(m, sha, policy, cell, n, T), **(header_changes or {}))
    log.write(path, header)


def rewrite_lines(path, fn):
    with gzip.open(path, "rt") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    lines = fn(lines)
    with gzip.open(path, "wt") as f:
        for x in lines:
            f.write(json.dumps(x, separators=(",", ":")) + "\n")


class EpisodeLogTests(unittest.TestCase):
    def test_terminal_sample_recorded_then_closed(self):
        log = ic.EpisodeLog(3, FEET, 0.02)
        for t in range(60):
            s = snap(3, t)
            if t == 59:
                s["base_height_m"] = [0.05, 0.32, 0.32]                 # state at the terminal step, pre-reset
            log.add(s, done=[t == 59] * 3)
        log.add(snap(3, 60), done=[False] * 3)                           # after close: ignored
        self.assertTrue(log.all_closed)
        rec = log.record(0)
        self.assertEqual(it.validate_episode(rec), 60)
        self.assertEqual(rec["base_height_m"][-1], 0.05)
        self.assertEqual(log.lengths(), [60, 60, 60])

    def test_write_read_roundtrip_with_footer(self):
        m = small_manifest(n=4)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s301__F040.jsonl.gz"
            write_run(p, m, "m", "s301", "F040", 4)
            header, eps, probs = ia.read_run(p, m, "m", "s301", "F040")
            self.assertEqual(len(eps), 4)
            self.assertEqual(probs, [])
            self.assertTrue(header["completed"])
            self.assertFalse(Path(str(p) + ".tmp").exists())
            self.assertEqual(ic.verify_written(p), [])
            rewrite_lines(p, lambda ls: ls[:-1])                         # truncated: collector would exit 1
            self.assertEqual(ic.verify_written(p), ["no footer"])


class PreResetCaptureTests(unittest.TestCase):
    """The step wrapper records the state captured inside reward computation, not the post-reset state."""

    def make(self, n=2, max_steps=3):
        col = object.__new__(ic.Collector)
        col.log, col.steps, col.max_steps = ic.EpisodeLog(n, FEET, 0.02), 0, max_steps
        col.act_buf = col.obs_buf = None
        col._pending, col._in_step, col.early_dones, col.started = None, False, 0, True
        col.finished = False
        col._finish = lambda env: setattr(col, "finished", True)
        col.sync = ic.SyncTrace([[0.5, 0.0, 0.0]] * n, max_steps)
        col.probe, col._pending_probe = ic.FrictionProbe(), None
        col._probe_snapshot = lambda env: ([[[3.0, 4.0, 50.0]] * 4] * n, [[[0.1, 0.0]] * 4] * n)
        col.sync.on_reset([0] * n)
        state = {"h": 0.32, "t": 0, "len": 0, "x": 0.0}
        env = types.SimpleNamespace(reward_manager=types.SimpleNamespace(compute=lambda *a, **k: "rew"))
        col._snapshot = lambda env: dict(snap(n, state["t"]), base_height_m=[state["h"]] * n)
        col._sync_snapshot = lambda env: {"len": [state["len"]] * n, "flag": [state["flag"]] * n,
                                          "pos": [[state["x"], 0.0, state["h"]]] * n, "cmd": [[0.5, 0.0, 0.0]] * n}
        env.episode_length_buf = types.SimpleNamespace(tolist=lambda: [state["len"]] * n)
        env.scene = {"robot": types.SimpleNamespace(data=types.SimpleNamespace(
            root_pos_w=types.SimpleNamespace(tolist=lambda: [[state["x"], 0.0, state["h"]]] * n)))}
        col._install_prereset_hook(env)
        return col, env, state

    def fake_step(self, env, state, n, done_at, fall_at=None):
        class Dones(list):
            def bool(self):
                return self

            def tolist(self):
                return list(self)

        def orig_step(wrapper, actions):
            state["t"] += 1
            state["len"] += 1                                           # Isaac Lab: length, terminations, rewards
            state["x"] += 0.01
            if fall_at is not None and state["t"] == fall_at:
                state["h"] = 0.05                                       # falls on this step
            done = state["t"] in done_at
            state["flag"] = done
            env.reward_manager.compute()                                # capture point (pre-reset)
            if done:
                state["h"], state["x"], state["len"] = 0.32, 0.0, 0     # the reset hides the fall
            return ("obs", "rew", Dones([done] * n), {})
        return orig_step

    def test_terminal_fall_not_hidden_by_reset(self):
        col, env, state = self.make(n=2, max_steps=3)
        wrapper = types.SimpleNamespace(unwrapped=env)
        orig = self.fake_step(env, state, 2, done_at={3}, fall_at=3)
        with self.assertRaises(KeyboardInterrupt):
            for _ in range(5):
                col.step(wrapper, orig, Act())
        self.assertTrue(col.finished)
        self.assertEqual(col.early_dones, 0)
        self.assertEqual(col.log.record(0)["base_height_m"][-1], 0.05)
        sync = col.sync.result(col.log.lengths())
        self.assertTrue(sync["ok"], sync["problems"])
        self.assertEqual(sync["reset_moved_robot"], 2)                  # the reset was real, and not recorded
        self.assertEqual(sync["terminal_steps"], [3])
        self.assertEqual(col.probe.sliding, 2 * 4 * 3)                  # every captured, open sample

    def test_early_done_counted(self):
        col, env, state = self.make(n=2, max_steps=5)
        wrapper = types.SimpleNamespace(unwrapped=env)
        orig = self.fake_step(env, state, 2, done_at={2})
        with self.assertRaises(KeyboardInterrupt):
            for _ in range(10):
                col.step(wrapper, orig, Act())
        self.assertEqual(col.early_dones, 2)
        self.assertFalse(col.sync.result(col.log.lengths())["ok"])      # ended at step 2, not 5

    def test_missing_capture_raises(self):
        col, env, state = self.make()
        wrapper = types.SimpleNamespace(unwrapped=env)
        with self.assertRaisesRegex(RuntimeError, "pre-reset"):
            col.step(wrapper, lambda w, a: ("o", "r", [False, False], {}), Act())


class SyncTraceTests(unittest.TestCase):
    """Each way the recorded trajectory could fall out of step with the environment is detected."""

    def run_trace(self, T=4, n=2, mutate=None):
        cmds = [[0.5, 0.0, 0.1]] * n
        tr = ic.SyncTrace(cmds, T)
        tr.on_reset([0] * n)
        open_ = [True] * n
        for k in range(1, T + 1):
            done = k == T
            ev = {"open": list(open_), "cap_len": [k] * n, "cap_flag": [done] * n,
                  "cap_pos": [[0.01 * k, 0.0, 0.3]] * n, "cap_cmd": [list(c) for c in cmds], "dones": [done] * n,
                  "post_len": [0 if done else k] * n,
                  "post_pos": [[0.0, 0.0, 0.3] if done else [0.01 * k, 0.0, 0.3]] * n}
            if mutate:
                mutate(k, ev)
            tr.on_step(k, ev["open"], ev["cap_len"], ev["cap_flag"], ev["cap_pos"], ev["cap_cmd"], ev["dones"],
                       ev["post_len"], ev["post_pos"])
            if done:
                open_ = [False] * n
        return tr.result([T] * n)

    def test_clean_trace_passes(self):
        r = self.run_trace()
        self.assertTrue(r["ok"], r["problems"])
        self.assertEqual(r["samples_checked"], 8)
        self.assertEqual(r["reset_moved_robot"], 2)

    def test_each_desynchronization_detected(self):
        def at(step, key, value):
            def m(k, ev):
                if k == step:
                    ev[key] = value
            return m
        cases = {
            "capture_length_not_consecutive": at(2, "cap_len", [3, 3]),        # a skipped or repeated sample
            "termination_flag_mismatch": at(4, "cap_flag", [False, False]),   # snapshot not at the terminal step
            "done_without_reset_after_capture": at(4, "post_len", [4, 4]),    # done but no reset after capture
            "state_changed_after_capture": at(2, "post_pos", [[9.0, 0.0, 0.3]] * 2),  # state moved after capture
            "command_mismatch": at(3, "cap_cmd", [[0.0, 0.0, 0.0]] * 2),      # command buffer not pinned
            "steps_without_physics_advance": at(3, "cap_pos", [[0.02, 0.0, 0.3]] * 2),  # step-2 state again
        }
        for key, mut in cases.items():
            r = self.run_trace(mutate=mut)
            self.assertFalse(r["ok"], key)
            self.assertTrue(any(p.startswith(key) for p in r["problems"]), (key, r["problems"]))

    def test_reset_length_and_end_step(self):
        tr = ic.SyncTrace([[0.5, 0.0, 0.0]], 3)
        tr.on_reset([7])                                                   # collector's reset did not take
        self.assertTrue(any("after the collector's reset" in p for p in tr.result([3])["problems"]))
        r = self.run_trace(T=4)
        tr = ic.SyncTrace([[0.5, 0.0, 0.1]] * 2, 5)                        # expected 5 steps, env ended at 4
        tr.on_reset([0, 0])
        for k in range(1, 5):
            d = k == 4
            tr.on_step(k, [True, True], [k, k], [d, d], [[0, 0, 0]] * 2, [[0.5, 0.0, 0.1]] * 2, [d, d],
                       [0 if d else k] * 2, [[1, 0, 0] if d else [0, 0, 0]] * 2)
        probs = tr.result([4, 4])["problems"]
        self.assertTrue(any("terminal transition at step 5" in p for p in probs), probs)
        self.assertTrue(any("expected exactly 5" in p for p in probs), probs)
        self.assertTrue(r["ok"])


class ContactMaterialTests(unittest.TestCase):
    GROUND = {"path": "/World/ground/terrain/physicsMaterial", "static": 1.0, "dynamic": 1.0, "mode": "multiply"}
    DEFAULT = {"path": "/physicsScene/defaultMaterial", "static": 1.0, "dynamic": 1.0, "mode": "multiply"}

    def attempt2(self, **kw):
        """The stage as S0 attempt 2 found it: 27 unbound robot shapes, ground with its own material."""
        e = {"ground_materials": [dict(self.GROUND)], "ground_collision_prims": 1, "ground_unbound_collision_prims": 0,
             "scene_default_material": dict(self.DEFAULT), "robot_prim": "/World/envs/env_0/Robot",
             "robot_collision_prims": 27, "robot_shapes": [], "robot_unbound_shapes": 27, "robot_materials_usd": [],
             "stage_materials": [dict(self.GROUND), dict(self.DEFAULT)],
             "robot_runtime_mode": {"observable": False, "reason": "no combine-mode query"}}
        e.update(kw)
        return e

    def test_combine_rule(self):
        self.assertAlmostEqual(ic.combine_friction(0.4, 1.0, "average", "multiply"), 0.4)
        self.assertAlmostEqual(ic.combine_friction(0.4, 1.0, "average", "average"), 0.7)
        self.assertAlmostEqual(ic.combine_friction(0.4, 1.0, "max", "multiply"), 1.0)
        self.assertAlmostEqual(ic.combine_friction(0.4, 0.8, "min", "average"), 0.4)
        with self.assertRaises(ValueError):
            ic.combine_friction(0.4, 1.0, "geometric", "multiply")

    def test_env0_path_resolution(self):
        """S0 attempt 1 failed here: Isaac Lab 3 writes the pattern as env_[^/]+, not env_.*."""
        paths = ["/World", "/World/envs", "/World/envs/env_0", "/World/envs/env_0/Robot",
                 "/World/envs/env_0/Robot/base", "/World/envs/env_1/Robot", "/World/envs/env_10/Robot"]
        for pat in ("/World/envs/env_[^/]+/Robot", "/World/envs/env_.*/Robot", "{ENV_REGEX_NS}/Robot"):
            self.assertEqual(ic.resolve_env0_path(pat, paths), "/World/envs/env_0/Robot", pat)
        with self.assertRaisesRegex(RuntimeError, "no prim under"):
            ic.resolve_env0_path("/World/envs/env_[^/]+/Dog", paths)

    def test_attempt2_stage_passes_with_recorded_limitation(self):
        for cid, mu in (("N080", 0.8), ("F040", 0.4), ("F110", 1.1)):
            e = self.attempt2()
            self.assertEqual(ic.contact_problems(e, ic.load_cell(MANIFEST, cid)), [], cid)
            self.assertEqual(e["effective_static"], [mu])
            self.assertEqual(e["effective_dynamic"], [round(0.75 * mu, 6)])
            self.assertEqual(set(e["robot_modes_considered"]), {"average", "multiply"})
            self.assertTrue(any("not observable" in x for x in e["limitations"]))

    def test_observed_runtime_mode_used_when_available(self):
        c = ic.load_cell(MANIFEST, "F040")
        e = self.attempt2(robot_runtime_mode={"observable": True, "modes": ["multiply"]})
        self.assertEqual(ic.contact_problems(e, c), [])
        self.assertEqual(e["robot_modes_considered"], {"multiply": "observed at runtime"})
        self.assertEqual(e["limitations"], [])
        e = self.attempt2(robot_runtime_mode={"observable": True, "modes": ["max"]})   # observed max: 1.0, not 0.4
        self.assertTrue(any("differs from the cell" in x for x in ic.contact_problems(e, c)))

    def test_failures_still_detected(self):
        c = ic.load_cell(MANIFEST, "F040")
        bad = {
            "ground on default": self.attempt2(ground_materials=[dict(self.DEFAULT)]),
            "ground unbound": self.attempt2(ground_unbound_collision_prims=1),
            "average ground": self.attempt2(ground_materials=[dict(self.GROUND, mode="average")]),
            "ground not 1.0": self.attempt2(ground_materials=[dict(self.GROUND, static=0.8, dynamic=0.8)]),
            "max on stage": self.attempt2(stage_materials=[dict(self.GROUND), dict(self.DEFAULT, mode="max")]),
            "modes not recorded": self.attempt2(stage_materials=None),
            "no robot prims": self.attempt2(robot_collision_prims=0, robot_unbound_shapes=0),
            "no ground": self.attempt2(ground_materials=[]),
            "bound max robot": self.attempt2(robot_unbound_shapes=0, robot_shapes=[
                {"prim": "x", "material": "/Robot/mat", "mode": "max"}]),
            "shared": self.attempt2(robot_unbound_shapes=0, robot_shapes=[
                {"prim": "x", "material": self.GROUND["path"], "mode": "average"}]),
            "error": {"error": "ImportError: pxr"},
        }
        for name, e in bad.items():
            self.assertNotEqual(ic.contact_problems(e, c), [], name)


class FrictionProbeTests(unittest.TestCase):
    def test_ratio_of_sliding_contacts_only(self):
        pr = ic.FrictionProbe()
        forces = [[[3.0, 4.0, 10.0], [3.0, 4.0, 50.0], [0.0, 6.0, 20.0], [1.0, 1.0, 5.0]]]   # 4 feet, 1 env
        vel = [[[0.2, 0.0], [0.01, 0.0], [0.0, 0.1], [0.3, 0.0]]]
        pr.add(forces, vel, [True])
        pr.add(forces, vel, [False])                                       # closed env: ignored
        r = pr.result(0.6)
        self.assertEqual(r["contact_samples"], 2)                          # fn > 10 N: feet 1 and 2
        self.assertEqual(r["sliding_samples"], 1)                          # foot 1 is not sliding
        self.assertAlmostEqual(r["median"], 0.3)                           # foot 2: 6 / 20
        self.assertIn("diagnostic only", r["role"])
        self.assertTrue(r["limitations"])

    def test_probe_is_not_a_validity_criterion(self):
        src = (HERE / "independent_analysis.py").read_text()
        self.assertNotIn("friction_probe", src)


class Act:
    def clone(self):
        return self


class RuntimeVerificationTests(unittest.TestCase):
    REMOVED = ("add_base_mass", "add_payload", "base_com", "push_robot", "base_external_force_torque")

    def obs(self, cell, **kw):
        k = cell["latency_steps"]
        o = {"active_terminations": ["time_out"], "active_events": ["physics_material", "reset_base"],
             "friction_static": [cell["static_friction"]] * 2, "friction_dynamic": [0.75 * cell["static_friction"]] * 2,
             "swept_motor": True, "motor_lag_steps_max": 0.0, "motor_strength": [1.0, 1.0],
             "latency": {"obs_buffer": cell["latency_side"] == "observation" and k > 0,
                         "action_buffer": cell["latency_side"] == "action" and k > 0}}
        o.update(kw)
        return o

    def test_all_cells_pass_when_runtime_matches(self):
        for c in MANIFEST["cells"]:
            self.assertEqual(ic.runtime_problems(self.obs(c), c, self.REMOVED), [], c["id"])

    def test_each_mismatch_detected(self):
        c = ic.load_cell(MANIFEST, "A1")
        cases = {"terminations": {"active_terminations": ["time_out", "base_contact"]},
                 "events": {"active_events": ["push_robot"]},
                 "static friction": {"friction_static": [0.8, 0.6]},
                 "dynamic friction": {"friction_dynamic": [0.8, 0.8]},
                 "swept DC motor": {"swept_motor": False},
                 "motor lag": {"motor_lag_steps_max": 4.0},
                 "motor strength": {"motor_strength": [0.9, 1.0]},
                 "latency buffers": {"latency": {"obs_buffer": True, "action_buffer": False}}}
        for word, kw in cases.items():
            probs = ic.runtime_problems(self.obs(c, **kw), c, self.REMOVED)
            self.assertTrue(any(word in p for p in probs), (word, probs))


def approved(stage_ids=("S1",)):
    m = copy.deepcopy(MANIFEST)
    m["status"] = "APPROVED"
    for st in m["stages"]:
        if st["id"] in stage_ids:
            st["status"] = "APPROVED"
    return m


class CollectorGuards(unittest.TestCase):
    def test_draft_manifest_refused(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "m.json"
            m.write_text(json.dumps(dict(MANIFEST, status="DRAFT")))
            with self.assertRaisesRegex(SystemExit, "not APPROVED"):
                ic.main(["--manifest", str(m), "--policy", "s301", "--cell", "F040", "--out", str(Path(d) / "o"),
                         "--stage", "S1", "--", "--task", MANIFEST["common_task"], "--checkpoint", "x.pt"])

    def test_stage_authorization(self):
        m = approved(stage_ids=("S1",))
        self.assertEqual(ic.authorize(m, "S1", "s301", "A1")["id"], "S1")
        with self.assertRaisesRegex(SystemExit, "not APPROVED"):
            ic.authorize(m, "S2", "s302", "A1")                          # stage not approved
        with self.assertRaisesRegex(SystemExit, "not a run of stage"):
            ic.authorize(m, "S1", "s302", "A1")                          # s302 is an S2 run
        with self.assertRaisesRegex(SystemExit, "not in the manifest"):
            ic.authorize(m, "S9", "s301", "A1")
        with self.assertRaisesRegex(SystemExit, "not APPROVED"):
            ic.authorize(dict(m, status="DRAFT"), "S1", "s301", "A1")    # manifest itself not approved
        self.assertEqual(ic.authorize(MANIFEST, "S0", "p0v5b_s101", "N080")["id"], "S0")  # committed: S0 only
        with self.assertRaisesRegex(SystemExit, "not APPROVED"):
            ic.authorize(MANIFEST, "S1", "s301", "A1")
        with self.assertRaisesRegex(SystemExit, "not a run of stage"):
            ic.authorize(MANIFEST, "S0", "p0v5b_s101", "F040")

    def test_stage_required(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            m = Path(d) / "m.json"
            m.write_text(json.dumps(approved()))
            with self.assertRaises(SystemExit):
                ic.main(["--manifest", str(m), "--policy", "s301", "--cell", "F040", "--out", str(Path(d) / "o"),
                         "--", "--task", MANIFEST["common_task"]])

    def test_wrong_task_and_num_envs_refused(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "m.json"
            m.write_text(json.dumps(approved()))
            base = ["--manifest", str(m), "--policy", "s301", "--cell", "F040", "--out", str(Path(d) / "o"),
                    "--stage", "S1", "--"]
            fixed = MANIFEST["play_args_fixed"]
            with self.assertRaisesRegex(SystemExit, "common task"):
                ic.main(base + ["--task", "Sim2Gate-Velocity-Flat-Go2-P2-R-slip-k0.1", "--checkpoint", "x.pt"] + fixed)
            with self.assertRaisesRegex(SystemExit, "fixed play arguments"):
                ic.main(base + ["--task", MANIFEST["common_task"], "--checkpoint", "x.pt", "physics=isaacsim_physx"])
            with self.assertRaisesRegex(SystemExit, "num_envs"):
                ic.main(base + ["--task", MANIFEST["common_task"], "--num_envs", "4"] + fixed)

    def test_cells_valid(self):
        for c in MANIFEST["cells"]:
            self.assertEqual(ic.load_cell(MANIFEST, c["id"])["id"], c["id"])
        with self.assertRaises(SystemExit):
            ic.load_cell(MANIFEST, "nope")

    def test_collector_imports_no_scoring_code(self):
        src = (HERE / "independent_collector.py").read_text()
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported |= {f"{node.module}.{a.name}" for a in node.names}
            elif isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
        allowed_sim2gate = {"sim2gate.checks", "sim2gate.checks.commands", "sim2gate.checks.isaaclab_conditions",
                            "sim2gate.checks.isaaclab_conditions.apply_profile",
                            "sim2gate.checks.isaaclab_conditions.install",
                            "sim2gate.checks.isaaclab_conditions.REMOVED_EVENTS"}
        self.assertEqual({m for m in imported if m.startswith("sim2gate")}, allowed_sim2gate)
        doc = tree.body[0]                                                # the docstring names what is NOT used
        self.assertIsInstance(doc, ast.Expr)
        code = "\n".join(src.splitlines()[doc.end_lineno:])
        for token in ("exploits_score", "layer1_limits", "transfer", "frozen_success", "check2_limit", "robustness"):
            self.assertNotIn(token, code)


class AnalysisTests(unittest.TestCase):
    N = 20

    def build(self, d, failures, overrides=None):
        """failures: {(policy, cell): fallen env count}; overrides: {(policy, cell): write_run kwargs}."""
        m = small_manifest(self.N)
        mp = Path(d) / "m.json"
        mp.write_text(json.dumps(m))
        sha = hashlib.sha256(mp.read_bytes()).hexdigest()
        for pid in m["policies"]:
            for c in m["cells"]:
                kw = dict((overrides or {}).get((pid, c["id"]), {}))
                kw.setdefault("T", 120)                                  # falls start at step 100
                write_run(Path(d) / f"{pid}__{c['id']}.jsonl.gz", m, sha, pid, c["id"], self.N,
                          fall_envs=failures.get((pid, c["id"]), 0), **kw)
        return m, sha, mp

    def test_manifestation_and_cross_symptoms(self):
        with tempfile.TemporaryDirectory() as d:
            m, sha, mp = self.build(d, {("s301", "F040"): 15, ("s311", "A2"): 18, ("s311", "N080"): 16,
                                        ("p0v5b_s101", "F040"): 1})
            res = ia.analyse(m, sha, d)
            self.assertEqual(res["status"], "COMPLETE")
            self.assertEqual(res["run_counts"], {"ok": 19 * 5, "invalid": 0, "missing": 0})
            P = res["policies"]
            self.assertTrue(P["s301"]["primary_truth_manifested"])           # 15/20 vs 1/100 controls
            self.assertFalse(P["s302"]["primary_truth_manifested"])
            self.assertTrue(P["s311"]["primary_truth_manifested"])           # action-delay primary cell
            self.assertEqual(P["s311"]["primary_cells"], ["A1", "A2"])
            self.assertEqual(P["s311"]["cross_symptom_cells"], ["N080"])
            self.assertIsNone(P["p0v5b_s101"]["primary_truth_manifested"])     # controls get no truth label
            self.assertEqual(res["pooled_controls"]["F040"], {"failures": 1, "valid": 100})
            self.assertAlmostEqual(P["s301"]["alpha_per_primary_cell"], m["truth_rules"]["alpha"] / 2)
            self.assertEqual(res["run_problems"], {})
            md = ia.to_markdown(res, [c["id"] for c in m["cells"]])
            self.assertIn("**Status: COMPLETE**", md)
            self.assertIn("| s301 | R-slip | MANIFESTED |", md)

    def test_missing_primary_run_makes_result_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            m, sha, mp = self.build(d, {("s301", "F040"): 15})
            (Path(d) / "s306__F045.jsonl.gz").unlink()
            res = ia.analyse(m, sha, d)
            self.assertEqual(res["status"], "INCOMPLETE")
            self.assertEqual(res["required_missing"], ["s306__F045"])
            self.assertEqual(res["policies"]["s306"]["cells"]["F045"]["status"], "missing")
            # no policy gets a truth label while the required set is incomplete, not even s301
            self.assertTrue(all(v["primary_truth_manifested"] is None for v in res["policies"].values()))

    def test_missing_secondary_run_does_not_block(self):
        with tempfile.TemporaryDirectory() as d:
            m, sha, mp = self.build(d, {})
            (Path(d) / "s306__N080.jsonl.gz").unlink()
            res = ia.analyse(m, sha, d)
            self.assertEqual(res["status"], "COMPLETE")
            self.assertEqual(res["run_counts"]["missing"], 1)
            self.assertFalse(res["policies"]["s306"]["primary_truth_manifested"])

    def test_invalid_control_run_excluded_from_pool(self):
        """A control run with early terminations must not enter the pooled reference."""
        with tempfile.TemporaryDirectory() as d:
            m, sha, mp = self.build(d, {("s301", "F040"): 15},
                                    {("p0v5b_s102", "F040"): {"header_changes": {"unexpected_early_dones": 3}}})
            res = ia.analyse(m, sha, d)
            self.assertEqual(res["status"], "INCOMPLETE")
            self.assertEqual(res["required_invalid"], ["p0v5b_s102__F040"])
            self.assertEqual(res["run_counts"]["invalid"], 1)
            self.assertIsNone(res["pooled_controls"]["F040"])
            self.assertIsNotNone(res["pooled_controls"]["F045"])
            self.assertIn("ended before the time limit", res["run_problems"]["p0v5b_s102__F040"][0])
            self.assertIsNone(res["policies"]["s301"]["primary_truth_manifested"])

    def test_each_invalidity_reason_detected(self):
        m = small_manifest(4)
        cases = {
            "not marked completed": "rewrite",
            "runtime verification": dict(header_changes={"runtime_checks": {"ok": False, "problems": ["x"]}}),
            "step synchronization": dict(header_changes={"sync_checks": {"ok": False, "problems": ["x"]}}),
            "checkpoint not confirmed": dict(header_changes={"checkpoint_matches_manifest": False}),
            "checkpoint digest": dict(header_changes={"checkpoint": {"sha256": "0" * 64}}),
            "cell definition": dict(header_changes={"cell": {"id": "F040", "static_friction": 0.5}}),
            "command set": dict(header_changes={"command_set_sha256": "x"}),
            "common task": dict(header_changes={"task": "Sim2Gate-Velocity-Flat-Go2-P2-R-slip-k0.1"}),
            "num_envs": dict(header_changes={"num_envs": 5}),
            "fixed play arguments": dict(header_changes={"play_args": ["--task", m["common_task"]]}),
            "ended before the time limit": dict(header_changes={"unexpected_early_dones": 1}),
            "not exactly episode_steps": dict(short_env=2),
        }
        with tempfile.TemporaryDirectory() as d:
            for word, kw in cases.items():
                p = Path(d) / f"{word.replace(' ', '_')}.jsonl.gz"
                if kw == "rewrite":                                      # writer always sets completed: true
                    write_run(p, m, "m", "s301", "F040", 4, T=60)
                    rewrite_lines(p, lambda ls: [{"header": dict(ls[0]["header"], completed=False)}] + ls[1:])
                else:
                    write_run(p, m, "m", "s301", "F040", 4, T=60, **kw)
                _, _, probs = ia.read_run(p, m, "m", "s301", "F040")
                self.assertTrue(any(word in x for x in probs), (word, probs))

    def test_truncated_tampered_and_bad_episode_runs_invalid(self):
        m = small_manifest(4)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s301__F040.jsonl.gz"
            write_run(p, m, "m", "s301", "F040", 4, T=60)
            rewrite_lines(p, lambda ls: ls[:-1])                         # footer lost
            self.assertIn("no footer (truncated or incomplete run)", ia.read_run(p, m, "m", "s301", "F040")[2])
            write_run(p, m, "m", "s301", "F040", 4, T=60)

            def tamper(ls):
                ls[1]["base_height_m"][10] = 0.05
                return ls
            rewrite_lines(p, tamper)
            self.assertIn("footer does not match the records", ia.read_run(p, m, "m", "s301", "F040")[2])
            write_run(p, m, "m", "s301", "F040", 4, T=60)

            def break_schema(ls):
                ls[2]["schema"] = "other"
                return ls
            rewrite_lines(p, break_schema)
            probs = ia.read_run(p, m, "m", "s301", "F040")[2]
            self.assertTrue(any("fail the schema" in x for x in probs), probs)
            p.write_bytes(b"not gzip")
            self.assertTrue(ia.read_run(p, m, "m", "s301", "F040")[2][0].startswith("unreadable"))

    def test_wrong_manifest_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            m, sha, mp = self.build(d, {})
            res = ia.analyse(m, "other", d)
            self.assertEqual(res["status"], "INCOMPLETE")
            self.assertTrue(all("different manifest" in v[0] for v in res["run_problems"].values()))

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            m, sha, mp = self.build(d, {})
            out = Path(d) / "out"
            out.mkdir()
            self.assertEqual(ia.main(["--manifest", str(mp), "--runs", d, "--out-json", str(out / "r.json")]), 0)
            (Path(d) / "s311__A1.jsonl.gz").unlink()
            self.assertEqual(ia.main(["--manifest", str(mp), "--runs", d, "--out-json", str(out / "r.json")]), 2)
            self.assertEqual(json.loads((out / "r.json").read_text())["status"], "INCOMPLETE")

    def test_analysis_imports_no_sim2gate(self):
        src = (HERE / "independent_analysis.py").read_text()
        mods = {(n.module or "").split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
        self.assertNotIn("sim2gate", mods)


class ManifestTests(unittest.TestCase):
    def test_manifest_consistent_with_verified_records(self):
        p1 = json.loads((HERE / "phase1_checkpoint_byte_verification_2026-10-30.json").read_text())["robots"]
        p2 = json.loads((HERE / "phase2_checkpoint_byte_verification_2026-10-30.json").read_text())["robots"]
        pol = MANIFEST["policies"]
        self.assertEqual(len(pol), 19)
        for pid, v in pol.items():
            src = p1[pid] if pid.startswith("p0v5b") else p2[pid]
            self.assertEqual(v["checkpoint_sha256"], src["checkpoint_sha256"], pid)
        self.assertEqual(MANIFEST["status"], "APPROVED")
        self.assertIn("Only stage S0", MANIFEST["approval"]["scope"])
        self.assertEqual(set(MANIFEST["truth_rules"]), set(it.REQUIRED_RULES))
        cs = json.loads((HERE / MANIFEST["command_set"]["path"]).read_text())
        canon = json.dumps([[float(x) for x in r] for r in cs["commands"]], separators=(",", ":"))
        self.assertEqual(hashlib.sha256(canon.encode()).hexdigest(), MANIFEST["command_set"]["sha256"])
        self.assertEqual(cs["sha256"], MANIFEST["command_set"]["sha256"])

    def test_delay_cells_and_roles(self):
        cells = {c["id"]: c for c in MANIFEST["cells"]}
        self.assertEqual(sorted(cells), ["A1", "A2", "F040", "F045", "F110", "N080", "O1", "O2"])
        self.assertEqual(MANIFEST["primary_cells"]["Phi-delay"], ["A1", "A2"])
        for c in ("A1", "A2"):
            self.assertEqual(cells[c]["latency_side"], "action")
        for c in ("O1", "O2"):
            self.assertEqual(cells[c]["latency_side"], "observation")
            self.assertIn(c, MANIFEST["secondary_cells"])
        primary = {c for cs in MANIFEST["primary_cells"].values() for c in cs}
        self.assertFalse(primary & set(MANIFEST["secondary_cells"]))
        self.assertTrue(all(c["latency_steps"] <= 2 for c in cells.values()))      # <= 40 ms; 60 ms cell dropped

    def test_stages_partition_runs_and_caps(self):
        st = {s["id"]: s for s in MANIFEST["stages"]}
        self.assertEqual(sorted(st), ["S0", "S1", "S2"])
        self.assertEqual(st["S0"]["status"], "APPROVED")                    # manager authorization 2026-10-30
        self.assertTrue(st["S0"]["approved_by"] and st["S0"]["approved_utc"])
        self.assertTrue(all(st[k]["status"] == "NOT_APPROVED" and st[k]["approved_by"] is None for k in ("S1", "S2")))
        everything = {(p, c["id"]) for p in MANIFEST["policies"] for c in MANIFEST["cells"]}
        s1 = [tuple(r) for r in st["S1"]["runs"]]
        s2 = [tuple(r) for r in st["S2"]["runs"]]
        self.assertEqual(len(s1) + len(s2), len(everything))
        self.assertEqual(set(s1) | set(s2), everything)
        self.assertFalse(set(s1) & set(s2))
        self.assertEqual(len(st["S0"]["runs"]), 1)
        self.assertTrue({tuple(r) for r in st["S0"]["runs"]} <= everything)
        self.assertFalse(st["S0"]["analysed"])
        pilot = {p for p, _ in s1}
        self.assertEqual({MANIFEST["policies"][p]["arm"] for p in pilot},
                         {"control", "R-slip", "Phi-slip-mild", "Phi-delay"})
        rate = MANIFEST["cost_model"]["rate_usd_per_hour"]
        for s in st.values():
            self.assertGreater(s["gpu_hour_cap"], 0)
            self.assertAlmostEqual(s["cost_cap_usd"], s["gpu_hour_cap"] * rate)
        tot = MANIFEST["total_caps"]
        self.assertAlmostEqual(tot["gpu_hours"], sum(s["gpu_hour_cap"] for s in st.values()))
        self.assertAlmostEqual(tot["cost_usd"], tot["gpu_hours"] * rate)
        self.assertEqual(tot["runs"], 1 + len(everything))
        per_run_h = MANIFEST["cost_model"]["per_run_minutes_assumed"] / 60
        for sid in ("S1", "S2"):
            self.assertGreaterEqual(st[sid]["gpu_hour_cap"], len(st[sid]["runs"]) * per_run_h)

    def test_power_table_reproduces(self):
        """Spot-check the manifest's power numbers with an exact computation (control rate 0, +0.15)."""
        rules = dict(MANIFEST["truth_rules"], alpha=MANIFEST["truth_rules"]["alpha"] / 2)
        n, nc = 128, 640
        pc, pp = 0.0, 0.15

        def pmf(k, n, p):
            return math.comb(n, k) * p ** k * (1 - p) ** (n - k)
        power = sum(pmf(k, n, pp) for k in range(n + 1)
                    if it.manifestation(k, n, 0, nc, rules)["manifested"])          # controls at 0 failures
        self.assertAlmostEqual(power, MANIFEST["power"]["table"]["control_rate_0.00"]["+0.15"], places=2)

    def test_cells_off_check1_friction_grid(self):
        grid = [0.8, 0.725, 0.65, 0.575, 0.5, 0.425, 0.35, 0.275, 0.2, 0.125]   # v2 envelope, for comparison only
        for c in MANIFEST["cells"]:
            if c["id"] in ("F040", "F045"):
                self.assertTrue(all(abs(c["static_friction"] - g) > 1e-6 for g in grid))


if __name__ == "__main__":
    unittest.main()
