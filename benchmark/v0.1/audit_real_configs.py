"""Gate 2: instantiate REAL Isaac Lab config classes; never launch a simulation.

Run inside an Isaac Lab-capable Python environment:
  PYTHONPATH=src python benchmark/v0.1/audit_real_configs.py

Writes JSON evidence only after all four configs instantiate. A missing Isaac Lab
runtime is BLOCKED, not PASS. No SimulationApp, gym.make, or training calls.
"""
import json
import math
import sys
from pathlib import Path

from sim2gate.v2 import tasks
from sim2gate.v2.phase2_arms import DYNAMIC_RATIO


def canonical(x):
    if isinstance(x, dict):
        return {str(k): canonical(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (tuple, list)):
        return [canonical(v) for v in x]
    if isinstance(x, (str, bool, int, float)) or x is None:
        return x
    if callable(x):
        return f"callable:{getattr(x, '__module__', '')}.{getattr(x, '__qualname__', type(x).__name__)}"
    if hasattr(x, "to_dict"):
        return canonical(x.to_dict())
    if hasattr(x, "__dict__"):
        return canonical(vars(x))
    return f"opaque:{type(x).__module__}.{type(x).__qualname__}"


def diffs(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for k in sorted(set(a) | set(b)):
            path = f"{prefix}.{k}" if prefix else k
            if k not in a or k not in b:
                result[path] = {"baseline": a.get(k, "<missing>"), "variant": b.get(k, "<missing>")}
            else:
                result.update(diffs(a[k], b[k], path))
        return result
    if a != b:
        return {prefix: {"baseline": a, "variant": b}}
    return {}


def approx_pair(actual, expected):
    return len(actual) == 2 and all(math.isclose(float(x), float(y), rel_tol=1e-10, abs_tol=1e-10)
                                    for x, y in zip(actual, expected))


def run():
    recipes = {
        "H0": ("Go2FlatP0EnvCfgV5b", "Go2FlatP0PPORunnerCfgV5b"),
        "T1": tasks._p2_names("R-slip", "k0.1"),
        "D1": tasks._p2_names("Phi-slip-mild", "mild"),
        "D2": tasks._p2_names("Phi-delay", "nodelay"),
    }
    configs = {}
    for label, (env_name, agent_name) in recipes.items():
        env = getattr(tasks, env_name)()
        agent = getattr(tasks, agent_name)()
        configs[label] = (env, agent)

    base_env, base_agent = configs["H0"]
    base_env_dict = canonical(base_env)
    base_agent_dict = canonical(base_agent)
    report = {"gate": "CPU Gate 2: real Isaac Lab configuration", "status": "FAIL", "recipes": {}}
    expected_env_paths = {
        "T1": {"rewards.feet_slide.weight"},
        "D1": {"events.physics_material.params.static_friction_range",
               "events.physics_material.params.dynamic_friction_range"},
        "D2": {"scene.robot.actuators.base_legs.min_delay",
               "scene.robot.actuators.base_legs.max_delay"},
    }
    checks_pass = True
    for label, (env, agent) in configs.items():
        ediff = diffs(base_env_dict, canonical(env))
        adiff = diffs(base_agent_dict, canonical(agent))
        allowed = expected_env_paths.get(label, set())
        # Some unchanged fields need not appear in the diff (e.g. min_delay already zero).
        unexpected_env = sorted(set(ediff) - allowed)
        unexpected_agent = sorted(set(adiff) - (set() if label == "H0" else {"experiment_name"}))
        pm = env.events.physics_material.params
        legs = env.scene.robot.actuators["base_legs"]
        weight = env.rewards.feet_slide.weight
        assertions = []
        if label == "T1":
            assertions = [math.isclose(weight, base_env.rewards.feet_slide.weight * 0.1, rel_tol=1e-10),
                          weight != 0]
        elif label == "D1":
            assertions = [approx_pair(pm["static_friction_range"], (0.3, 0.6)),
                          approx_pair(pm["dynamic_friction_range"],
                                      (0.3 * DYNAMIC_RATIO, 0.6 * DYNAMIC_RATIO)),
                          (pm["static_friction_range"][1] - pm["static_friction_range"][0]) <
                          (base_env.events.physics_material.params["static_friction_range"][1] -
                           base_env.events.physics_material.params["static_friction_range"][0])]
        elif label == "D2":
            assertions = [legs.min_delay == 0, legs.max_delay == 0,
                          base_env.scene.robot.actuators["base_legs"].max_delay > 0]
        passed = not unexpected_env and not unexpected_agent and all(assertions) and (label == "H0" or bool(ediff))
        checks_pass &= passed
        report["recipes"][label] = {
            "status": "PASS" if passed else "FAIL",
            "env_diff_paths": sorted(ediff), "agent_diff_paths": sorted(adiff),
            "unexpected_env_paths": unexpected_env, "unexpected_agent_paths": unexpected_agent,
            "assertions": assertions,
            "observed": {"feet_slide_weight": weight,
                         "static_friction": list(pm["static_friction_range"]),
                         "dynamic_friction": list(pm["dynamic_friction_range"]),
                         "min_delay": legs.min_delay, "max_delay": legs.max_delay},
        }
    report["status"] = "PASS" if checks_pass else "FAIL"
    out = Path("benchmark/v0.1/GATE2_REAL_CONFIG_REPORT.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved {out}")
    return 0 if checks_pass else 1


if __name__ == "__main__":
    try:
        sys.exit(run())
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        print(f"BLOCKED: real Isaac Lab config could not instantiate: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
