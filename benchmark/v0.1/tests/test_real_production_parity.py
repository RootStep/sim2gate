"""Unmocked synthetic scored-case parity and label-invariance regression tests.

Runs unchanged production Check 1/2/5 scoring functions in a CPU-only harness.
No held-out keys, policies, training or real measurements are used.
"""
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark" / "v0.1"))
sys.path.insert(0, str(ROOT / "src"))
package = types.ModuleType("sim2gate")
package.__path__ = [str(ROOT / "src" / "sim2gate")]
checks = types.ModuleType("sim2gate.checks")
checks.__path__ = [str(ROOT / "src" / "sim2gate" / "checks")]
measurement = types.ModuleType("sim2gate.checks.isaaclab_exploits")
measurement.CALIBRATED_METRICS = (
    "foot_slip_p95_mps", "ground_penetration_m", "contact_force_p99_bw",
    "action_jitter_high_freq_power_share", "actuator_limit_step_share")
measurement.MEASUREMENT_KEYS = (
    "dt", "contact_force_n", "foot_radius_m", "penetration_tol_m",
    "jitter_window", "flat_ground", "measurement_version", "foot_geometry")
sys.modules.update({"sim2gate": package, "sim2gate.checks": checks,
                    "sim2gate.checks.isaaclab_exploits": measurement})
from sim2gate.checks import robustness, transfer, exploits_score
from check15_sanitized_adapter import export_check1, export_check5
from check2_sanitized_adapter import export_check2

SECRET = "PRIVATE_SEALED_LABEL_CANARY"


def ep(error=0.05, fell=False, group=0):
    return {"mean_error_mps": error, "fell": fell, "group": group}


class RealProductionParity(unittest.TestCase):
    def test_check1_scored_cliff_and_label_invariance(self):
        groups = robustness.conditions()
        raw = {"grid": robustness.grid_definition(), "groups": groups,
               "profile": "nominal", "command_set_sha256": "commands",
               "episode_steps": 100,
               "checkpoint": {"sha256": "PRIVATE_CHECKPOINT"},
               "episodes": [ep(group=g["group"], error=0.2 if
                            g["param"] == "ground_friction" and g["index"] >= 3 else 0.05)
                            for g in groups for _ in range(64)]}
        calibration = {"threshold_mps": 0.1, "profile": "nominal",
                       "command_set_sha256": "commands", "episode_steps": 100}
        direct = robustness.score(raw, calibration)
        self.assertEqual(direct["status"], "scored")
        self.assertTrue(direct["params"]["ground_friction"]["flagged"])
        self.assertTrue(direct["flagged"])
        projected = export_check1("opaque01", raw, calibration)
        self.assertEqual(projected["flagged"], direct["flagged"])
        self.assertEqual(projected["metrics"],
                         {k: v["flagged"] for k, v in direct["params"].items()})
        for label in (True, False):
            changed = {**raw, "ground_truth": label, "recipe": SECRET}
            self.assertEqual(projected, export_check1("opaque01", changed, calibration))
        self.assertNotIn(SECRET, json.dumps(projected))
        invalid = export_check1("opaque01", raw, {**calibration, "profile": "different"})
        self.assertEqual(invalid["status"], "invalid")
        self.assertIsNone(invalid["flagged"])
        self.assertTrue(all(x is None for x in invalid["metrics"].values()))

    def test_check5_scored_both_risks_and_label_invariance(self):
        common = {"task": "go2", "checkpoint_sha256": "PRIVATE_CHECKPOINT",
                  "command_set_sha256": "commands", "step_dt": 0.02,
                  "decimation": 4, "sim_dt": 0.005, "seed": 1, "num_envs": 64,
                  "episode_steps": 100, "profile": "nominal",
                  "env_cfg_sha256": "cfg", "joint_names": ["j1"],
                  "body_names": ["b1"], "conditions": {"varied": False}}
        ordering = list(transfer.ORDERING) + list(transfer.FIXED_OVERRIDES)
        physx = {**common, "overrides": [transfer.PHYSX, *ordering],
                 "physics": {"manager": "physx"}, "episodes": [ep(0.05) for _ in range(64)]}
        newton = {**common, "overrides": [transfer.NEWTON, *ordering],
                  "physics": {"manager": "newton", "solver_cfg": {"iterations": 4},
                              "versions": {"engine": "synthetic"}},
                  "episodes": [ep(0.2, fell=i < 20) for i in range(64)]}
        calibration = {"threshold_mps": 0.1, "profile": "nominal",
                       "command_set_sha256": "commands", "episode_steps": 100}
        direct = transfer.score(physx, newton, calibration)
        self.assertEqual(direct["status"], "scored", direct["reasons"])
        self.assertEqual({r["metric"] for r in direct["risks"]},
                         {"newton_fall_free_rate", "error_ratio"})
        projected = export_check5("opaque02", physx, newton, calibration)
        self.assertEqual(projected["flagged"], direct["flagged"])
        self.assertEqual(projected["metrics"],
                         {name: name in {r["metric"] for r in direct["risks"]}
                          for name in projected["metrics"]})
        self.assertEqual(projected, export_check5(
            "opaque02", {**physx, "ground_truth": SECRET},
            {**newton, "recipe": SECRET}, calibration))
        self.assertNotIn(SECRET, json.dumps(projected))
        invalid = export_check5("opaque02", physx,
                                {**newton, "checkpoint_sha256": "mismatch"}, calibration)
        self.assertEqual(invalid["status"], "invalid")
        self.assertIsNone(invalid["flagged"])
        self.assertTrue(all(v is None for v in invalid["metrics"].values()))

    def test_check2_scored_and_label_invariance(self):
        config = {k: 1 for k in measurement.MEASUREMENT_KEYS}
        thresholds = {k: 0.1 for k in measurement.CALIBRATED_METRICS}
        calibration = {"measurement_config": config,
                       "contact_force_samples_per_step": 4,
                       "physics_context": {"physics_handles_decimation": True,
                                           "overrides": ["fixed"]},
                       "command_set": {"sha256": "commands"},
                       "sources": [{"env_steps": 100}],
                       "thresholds": thresholds}
        report = {"checkpoint": {"sha256": "PRIVATE_CHECKPOINT"},
                  "config": config,
                  "contact_force_substeps": {"samples_per_control_step": 4},
                  "physics_context": calibration["physics_context"],
                  "commands": {"fixed": True, "sha256": "commands"},
                  "env_steps": 100,
                  "coverage": {"torque": 1, "joint_velocity": 1},
                  "metrics": {**{k: 0.05 for k in thresholds},
                              "foot_slip_p95_mps": 0.2}}
        direct = exploits_score.score(report, calibration)
        self.assertEqual(direct["status"], "scored", direct["reasons"])
        projected = export_check2("opaque03", report, calibration)
        self.assertEqual(projected["flagged"], direct["flagged"])
        self.assertEqual(projected["metrics"],
                         {k: v["flagged"] for k, v in direct["metrics"].items()})
        self.assertEqual(projected, export_check2(
            "opaque03", {**report, "ground_truth": SECRET, "recipe": SECRET}, calibration))
        self.assertNotIn(SECRET, json.dumps(projected))
        invalid = export_check2("opaque03", {**report, "env_steps": 99}, calibration)
        self.assertEqual(invalid["status"], "invalid")
        self.assertIsNone(invalid["flagged"])
        self.assertTrue(all(v is None for v in invalid["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
