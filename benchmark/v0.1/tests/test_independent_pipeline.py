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


def write_run(path, policy, cell, n, fall_envs=0, T=200, manifest_sha="m", slide=0.0):
    log = ic.EpisodeLog(n, FEET, 0.02)
    for t in range(T):
        s = snap(n, t, slide=slide)
        if t >= 100:
            s["base_height_m"] = [0.05 if i < fall_envs else 0.32 for i in range(n)]
        log.add(s, done=[t == T - 1] * n)
    header = {"manifest_sha256": manifest_sha, "policy_id": policy, "cell": {"id": cell},
              "checkpoint_matches_manifest": True, "num_envs": n}
    log.write(path, header)


class EpisodeLogTests(unittest.TestCase):
    def test_records_valid_and_final_post_reset_step_dropped(self):
        log = ic.EpisodeLog(3, FEET, 0.02)
        for t in range(60):
            log.add(snap(3, t), done=[t == 59, t == 59, t == 59])
        self.assertTrue(log.all_closed)
        rec = log.record(0)
        self.assertEqual(it.validate_episode(rec), 59)
        self.assertEqual(set(rec["feet"]), set(FEET))

    def test_write_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl.gz"
            write_run(p, "s301", "F040", 4)
            header, eps, probs = ia.read_run(p, "m")
            self.assertEqual(len(eps), 4)
            self.assertEqual(probs, [])
            self.assertFalse(Path(str(p) + ".tmp").exists())


class CollectorGuards(unittest.TestCase):
    def test_draft_manifest_refused(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "m.json"
            m.write_text(json.dumps(MANIFEST))
            with self.assertRaisesRegex(SystemExit, "not APPROVED"):
                ic.main(["--manifest", str(m), "--policy", "s301", "--cell", "F040", "--out", str(Path(d) / "o"),
                         "--", "--task", MANIFEST["common_task"], "--checkpoint", "x.pt"])

    def test_wrong_task_and_num_envs_refused(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "m.json"
            m.write_text(json.dumps(dict(MANIFEST, status="APPROVED")))
            base = ["--manifest", str(m), "--policy", "s301", "--cell", "F040", "--out", str(Path(d) / "o"), "--"]
            with self.assertRaisesRegex(SystemExit, "common task"):
                ic.main(base + ["--task", "Sim2Gate-Velocity-Flat-Go2-P2-R-slip-k0.1", "--checkpoint", "x.pt"])
            with self.assertRaisesRegex(SystemExit, "num_envs"):
                ic.main(base + ["--task", MANIFEST["common_task"], "--num_envs", "4"])

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
                            "sim2gate.checks.isaaclab_conditions.install"}
        self.assertEqual({m for m in imported if m.startswith("sim2gate")}, allowed_sim2gate)
        doc = tree.body[0]                                                # the docstring names what is NOT used
        self.assertIsInstance(doc, ast.Expr)
        code = "\n".join(src.splitlines()[doc.end_lineno:])
        for token in ("exploits_score", "layer1_limits", "transfer", "frozen_success", "check2_limit", "robustness"):
            self.assertNotIn(token, code)


class AnalysisTests(unittest.TestCase):
    def manifest(self):
        m = copy.deepcopy(MANIFEST)
        m["cells"] = [c for c in m["cells"] if c["id"] in ("N080", "F040", "F045", "L1", "L2")]
        return m

    def build(self, d, failures):
        """failures: {(policy, cell): fallen env count}; every run has 20 episodes."""
        m = self.manifest()
        mp = Path(d) / "m.json"
        mp.write_text(json.dumps(m))
        sha = hashlib.sha256(mp.read_bytes()).hexdigest()
        for pid in m["policies"]:
            for c in m["cells"]:
                write_run(Path(d) / f"{pid}__{c['id']}.jsonl.gz", pid, c["id"], 20,
                          fall_envs=failures.get((pid, c["id"]), 0), manifest_sha=sha)
        return m, sha, mp

    def test_manifestation_and_cross_symptoms(self):
        with tempfile.TemporaryDirectory() as d:
            m, sha, mp = self.build(d, {("s301", "F040"): 15, ("s311", "L2"): 18, ("s311", "N080"): 16,
                                        ("p0v5b_s101", "F040"): 1})
            res = ia.analyse(m, sha, d)
            P = res["policies"]
            self.assertTrue(P["s301"]["primary_truth_manifested"])           # 15/20 vs 1/100 controls
            self.assertFalse(P["s302"]["primary_truth_manifested"])
            self.assertTrue(P["s311"]["primary_truth_manifested"])
            self.assertEqual(P["s311"]["cross_symptom_cells"], ["N080"])
            self.assertIsNone(P["p0v5b_s101"]["primary_truth_manifested"])     # controls get no truth label
            self.assertEqual(res["pooled_controls"]["F040"], {"failures": 1, "valid": 100})
            self.assertAlmostEqual(P["s301"]["alpha_per_primary_cell"], m["truth_rules"]["alpha"] / 2)
            self.assertEqual(res["run_problems"], {})
            md = ia.to_markdown(res, [c["id"] for c in m["cells"]])
            self.assertIn("| s301 | R-slip | MANIFESTED |", md)

    def test_missing_run_not_imputed(self):
        with tempfile.TemporaryDirectory() as d:
            m, sha, mp = self.build(d, {})
            (Path(d) / "s306__F045.jsonl.gz").unlink()
            res = ia.analyse(m, sha, d)
            self.assertIsNone(res["policies"]["s306"]["primary_truth_manifested"])
            self.assertEqual(res["policies"]["s306"]["missing_primary_cells"], ["F045"])
            (Path(d) / "p0v5b_s102__L1.jsonl.gz").unlink()                       # a missing control run
            res = ia.analyse(m, sha, d)
            self.assertIsNone(res["pooled_controls"]["L1"])
            self.assertIsNone(res["policies"]["s311"]["primary_truth_manifested"])

    def test_wrong_manifest_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            m, sha, mp = self.build(d, {})
            res = ia.analyse(m, "other", d)
            self.assertTrue(all("different manifest" in v[0] for v in res["run_problems"].values()))

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
        self.assertEqual(MANIFEST["status"], "DRAFT")
        self.assertEqual(set(MANIFEST["truth_rules"]), set(it.REQUIRED_RULES))
        cs = json.loads((HERE / MANIFEST["command_set"]["path"]).read_text())
        canon = json.dumps([[float(x) for x in r] for r in cs["commands"]], separators=(",", ":"))
        self.assertEqual(hashlib.sha256(canon.encode()).hexdigest(), MANIFEST["command_set"]["sha256"])
        self.assertEqual(cs["sha256"], MANIFEST["command_set"]["sha256"])

    def test_cells_off_check1_friction_grid(self):
        grid = [0.8, 0.725, 0.65, 0.575, 0.5, 0.425, 0.35, 0.275, 0.2, 0.125]   # v2 envelope, for comparison only
        for c in MANIFEST["cells"]:
            if c["id"] in ("F040", "F045"):
                self.assertTrue(all(abs(c["static_friction"] - g) > 1e-6 for g in grid))


if __name__ == "__main__":
    unittest.main()
