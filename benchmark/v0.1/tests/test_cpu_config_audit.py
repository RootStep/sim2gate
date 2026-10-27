"""CPU-only structural audit for proposed benchmark variants.

Run with standard Python, no Isaac Lab or GPU required:
  PYTHONPATH=src python -m unittest discover -s benchmark/v0.1/tests -p 'test_cpu_config_audit.py' -v

Tests the existing pure-Python phase2_arms.apply against minimal stand-in configs.
Does NOT prove real Isaac Lab config instantiation, simulator sampling, or training outcomes.
"""
import copy
import unittest
from types import SimpleNamespace

from sim2gate.v2 import phase2_arms


def baseline():
    return SimpleNamespace(
        rewards=SimpleNamespace(feet_slide=SimpleNamespace(weight=-0.25)),
        events=SimpleNamespace(physics_material=SimpleNamespace(params={
            "static_friction_range": (0.35, 1.2),
            "dynamic_friction_range": (0.2625, 0.9),
            "make_consistent": True,
        })),
        scene=SimpleNamespace(robot=SimpleNamespace(actuators={
            "base_legs": SimpleNamespace(min_delay=0, max_delay=6)
        })),
    )


def snapshot(cfg):
    return {
        "reward.feet_slide.weight": cfg.rewards.feet_slide.weight,
        "friction.static": cfg.events.physics_material.params["static_friction_range"],
        "friction.dynamic": cfg.events.physics_material.params["dynamic_friction_range"],
        "friction.consistent": cfg.events.physics_material.params["make_consistent"],
        "actuator.min_delay": cfg.scene.robot.actuators["base_legs"].min_delay,
        "actuator.max_delay": cfg.scene.robot.actuators["base_legs"].max_delay,
    }


def changed(before, after):
    return {key for key in before if before[key] != after[key]}


class TestCpuConfigAudit(unittest.TestCase):
    def test_h0_unchanged(self):
        original = baseline()
        self.assertEqual(snapshot(original), snapshot(copy.deepcopy(original)))

    def test_t1_reduced_slip_only(self):
        cfg = baseline()
        before = snapshot(cfg)
        phase2_arms.apply(cfg, "R-slip", "k0.1")
        after = snapshot(cfg)
        self.assertEqual(changed(before, after), {"reward.feet_slide.weight"})
        self.assertAlmostEqual(after["reward.feet_slide.weight"], before["reward.feet_slide.weight"] * 0.1)
        self.assertNotEqual(after["reward.feet_slide.weight"], 0)

    def test_d1_friction_only(self):
        cfg = baseline()
        before = snapshot(cfg)
        phase2_arms.apply(cfg, "Phi-slip-mild", "mild")
        after = snapshot(cfg)
        self.assertEqual(changed(before, after), {"friction.static", "friction.dynamic"})
        self.assertEqual(after["friction.static"], (0.3, 0.6))
        self.assertEqual(after["friction.dynamic"], (0.225, 0.44999999999999996))
        self.assertLess(after["friction.static"][1]-after["friction.static"][0], before["friction.static"][1]-before["friction.static"][0])

    def test_d2_no_delay_only(self):
        cfg = baseline()
        before = snapshot(cfg)
        phase2_arms.apply(cfg, "Phi-delay", "nodelay")
        after = snapshot(cfg)
        self.assertEqual(changed(before, after), {"actuator.max_delay"})
        self.assertEqual(after["actuator.max_delay"], 0)

    def test_missing_reward_term_refused(self):
        cfg = baseline()
        cfg.rewards.feet_slide = None
        with self.assertRaises(ValueError):
            phase2_arms.apply(cfg, "R-slip", "k0.1")

    def test_missing_delay_actuator_refused(self):
        cfg = baseline()
        del cfg.scene.robot.actuators["base_legs"].max_delay
        with self.assertRaises(RuntimeError):
            phase2_arms.apply(cfg, "Phi-delay", "nodelay")

    def test_no_zero_reward_baseline(self):
        cfg = baseline()
        cfg.rewards.feet_slide.weight = 0
        with self.assertRaises(ValueError):
            phase2_arms.apply(cfg, "R-slip", "k0.1")


if __name__ == "__main__":
    unittest.main()
