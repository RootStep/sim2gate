"""Unit tests for the independent behavioural-truth library (gate G1). Synthetic trajectories only."""
import ast
import copy
import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import independent_truth as it

R = dict(it.DEFAULT_RULES)
DT = 0.02


def episode(T=200, height=0.32, tilt_deg=0.0, cmd=(0.5, 0.0), vel=(0.5, 0.0), feet=None, base_speed=0.5):
    g = [math.sin(math.radians(tilt_deg)), 0.0, -math.cos(math.radians(tilt_deg))]
    ep = {"schema": it.SCHEMA_VERSION, "dt": DT,
          "base_height_m": [height] * T, "projected_gravity_b": [list(g) for _ in range(T)],
          "base_lin_vel_b_xy": [list(vel) for _ in range(T)], "cmd_vel_xy": [list(cmd) for _ in range(T)],
          "base_pos_w_xy": [[base_speed * DT * t, 0.0] for t in range(T)],
          "feet": feet or {"FL": stationary_foot(T)}}
    return ep


def stationary_foot(T, period=20, stance=12, x=0.0):
    """Alternating stance (force 60 N) and swing (0 N); position fixed during stance, jumps between stances."""
    force, pos = [], []
    for t in range(T):
        on = (t % period) < stance
        force.append(60.0 if on else 0.0)
        pos.append([x + 0.1 * (t // period), 0.0])
    return {"contact_force_n": force, "pos_w_xy": pos}


def sliding_foot(T, speed, period=20, stance=12):
    f = stationary_foot(T, period, stance)
    f["pos_w_xy"] = [[0.1 * (t // period) + (speed * DT * (t % period) if (t % period) < stance else 0.0), 0.0]
                     for t in range(T)]
    return f


class Schema(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(it.validate_episode(episode()), 200)

    def test_rejects(self):
        cases = []
        e = episode(); e["schema"] = "other"; cases.append(e)
        e = episode(); e["dt"] = 0; cases.append(e)
        e = episode(); e["cmd_vel_xy"].pop(); cases.append(e)
        e = episode(); e["base_height_m"][3] = float("nan"); cases.append(e)
        e = episode(); e["feet"]["FL"]["pos_w_xy"][0] = [0.0]; cases.append(e)
        e = episode(); e["feet"] = {}; cases.append(e)
        e = episode(); e["projected_gravity_b"][0] = [0.0, 0.0]; cases.append(e)
        for c in cases:
            with self.subTest(), self.assertRaises(it.SchemaError):
                it.validate_episode(c)

    def test_rules_required(self):
        with self.assertRaises(ValueError):
            it.episode_outcome(episode(), {k: v for k, v in R.items() if k != "alpha"})


class FallsAndTracking(unittest.TestCase):
    def test_tilt(self):
        self.assertAlmostEqual(it.tilt_deg([0, 0, -9.81]), 0.0)
        self.assertAlmostEqual(it.tilt_deg([0, 9.81, 0]), 90.0)
        self.assertAlmostEqual(it.tilt_deg([0, 0, 9.81]), 180.0)

    def test_upright_no_fall(self):
        o = it.episode_outcome(episode(), R)
        self.assertFalse(o["fell"]); self.assertFalse(o["failure"]); self.assertAlmostEqual(o["tracking_error_mps"], 0.0)

    def test_sustained_low_base_is_fall(self):
        e = episode()
        for t in range(120, 200):
            e["base_height_m"][t] = 0.1
        o = it.episode_outcome(e, R)
        self.assertTrue(o["fell"]); self.assertEqual(o["fall_step"], 120); self.assertTrue(o["failure"])

    def test_brief_dip_is_not_fall(self):
        e = episode()
        for t in range(100, 100 + R["fall_persist_steps"] - 1):
            e["base_height_m"][t] = 0.1
        self.assertFalse(it.episode_outcome(e, R)["fell"])

    def test_tilt_fall(self):
        e = episode()
        g = [math.sin(math.radians(75)), 0.0, -math.cos(math.radians(75))]
        for t in range(150, 200):
            e["projected_gravity_b"][t] = list(g)
        self.assertEqual(it.episode_outcome(e, R)["fall_step"], 150)

    def test_tracking_loss_and_warmup(self):
        o = it.episode_outcome(episode(vel=(0.2, 0.0)), R)            # error 0.3 > 0.25
        self.assertAlmostEqual(o["tracking_error_mps"], 0.3); self.assertTrue(o["tracking_loss"]); self.assertTrue(o["failure"])
        e = episode()
        for t in range(R["tracking_warmup_steps"]):                   # large error only during warm-up
            e["base_lin_vel_b_xy"][t] = [0.0, 0.0]
        self.assertAlmostEqual(it.episode_outcome(e, R)["tracking_error_mps"], 0.0)

    def test_tracking_stops_at_fall(self):
        e = episode()
        for t in range(120, 200):
            e["base_height_m"][t] = 0.1
            e["base_lin_vel_b_xy"][t] = [5.0, 5.0]                    # post-fall motion excluded
        self.assertAlmostEqual(it.episode_outcome(e, R)["tracking_error_mps"], 0.0)


class Slip(unittest.TestCase):
    def test_stance_phases_trimmed_and_short_contacts_dropped(self):
        force = [0.0] * 5 + [50.0] * 12 + [0.0] * 3 + [50.0] * 2 + [0.0] * 3
        self.assertEqual(it.stance_phases(force, R), [(6, 16)])        # 12-step stance trimmed by 1; 2-step dropped

    def test_force_threshold(self):
        self.assertEqual(it.stance_phases([R["contact_force_n"]] * 10, R), [])  # equal is not above

    def test_stationary_foot_has_zero_slip(self):
        s = it.stance_slip(episode(feet={"FL": stationary_foot(200)}), R)
        self.assertEqual(s["stances"], 10); self.assertAlmostEqual(s["total_m"], 0.0); self.assertAlmostEqual(s["median_m"], 0.0)

    def test_sliding_foot_known_path(self):
        speed = 0.5
        s = it.stance_slip(episode(feet={"FL": sliding_foot(200, speed)}), R)
        expected = speed * DT * (12 - 2 * R["stance_trim_steps"] - 1)   # 10 trimmed samples -> 9 intervals
        self.assertEqual(s["stances"], 10)
        self.assertAlmostEqual(s["median_m"], expected); self.assertAlmostEqual(s["p90_m"], expected)
        self.assertAlmostEqual(s["slip_ratio"], s["total_m"] / s["base_travel_m"])

    def test_swing_motion_not_counted(self):
        f = stationary_foot(200)
        for t in range(200):
            if (t % 20) >= 12:
                f["pos_w_xy"][t] = [99.0 + t, 0.0]                         # large swing motion, no contact
        self.assertAlmostEqual(it.stance_slip(episode(feet={"FL": f}), R)["total_m"], 0.0)

    def test_slip_stops_at_end(self):
        s = it.stance_slip(episode(feet={"FL": sliding_foot(200, 0.5)}), R, end=100)
        self.assertEqual(s["stances"], 5)

    def test_no_travel_ratio_none(self):
        self.assertIsNone(it.stance_slip(episode(base_speed=0.0), R)["slip_ratio"])


class Manifestation(unittest.TestCase):
    def test_fisher_known_values(self):
        self.assertAlmostEqual(it.fisher_one_sided_greater(3, 1, 1, 3), 17 / 70, places=12)   # tea tasting
        self.assertAlmostEqual(it.fisher_one_sided_greater(5, 0, 0, 5), 1 / 252, places=12)
        self.assertAlmostEqual(it.fisher_one_sided_greater(0, 5, 5, 0), 1.0, places=12)
        self.assertAlmostEqual(it.fisher_one_sided_greater(0, 10, 0, 10), 1.0, places=12)

    def test_fisher_rejects_bad_counts(self):
        for bad in ((-1, 1, 1, 1), (1.5, 1, 1, 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                it.fisher_one_sided_greater(*bad)

    def test_manifestation_rule(self):
        self.assertTrue(it.manifestation(40, 64, 2, 320, R)["manifested"])
        self.assertFalse(it.manifestation(3, 64, 10, 320, R)["manifested"])   # no excess
        weak = it.manifestation(30, 640, 1, 3200, R)                            # significant but < 0.10 apart
        self.assertLess(weak["p_value"], R["alpha"]); self.assertFalse(weak["manifested"])
        with self.assertRaises(ValueError):
            it.manifestation(5, 4, 0, 10, R)


class Latency(unittest.TestCase):
    def test_zero_delay_passthrough(self):
        b = it.LatencyBuffer(0)
        self.assertEqual([b(x) for x in range(5)], [0, 1, 2, 3, 4])

    def test_two_step_delay(self):
        b = it.LatencyBuffer(2)
        self.assertEqual([b(x) for x in range(6)], [0, 0, 0, 1, 2, 3])
        b.reset()
        self.assertEqual(b(9), 9)

    def test_invalid(self):
        for s in (-1, 1.5):
            with self.subTest(s=s), self.assertRaises(ValueError):
                it.LatencyBuffer(s)


class Independence(unittest.TestCase):
    """G1: the truth code must not reuse Sim2Gate's measurement code or calibrated limits."""

    SRC = (HERE / "independent_truth.py").read_text()

    def test_imports_are_standard_library_only(self):
        mods = set()
        for node in ast.walk(ast.parse(self.SRC)):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        self.assertEqual(mods, {"math"})

    def test_no_sim2gate_thresholds_or_files(self):
        for token in ("1.054", "0.1187", "0.1124", "0.11848", "check2_limit", "frozen_success", "layer1_limits",
                      "exploits_score", "robustness", "isaaclab_exploits"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.SRC)

    def test_default_rules_are_marked_proposals(self):
        self.assertIn("PROPOSED", self.SRC)
        self.assertEqual(set(it.REQUIRED_RULES), set(it.DEFAULT_RULES))


if __name__ == "__main__":
    unittest.main()
