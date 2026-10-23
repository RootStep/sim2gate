"""v2 phase 2 arms and their candidate settings (docs/v2/PHASE2_PROTOCOL.md, part A).

Every arm is P0 v5b with one change. Every arm keeps P0 v5b's reward terms (same names, same order, same logging);
reward arms rescale an existing term's weight and never set it to zero. Physics arms change a training range.
Candidates are tried in order in the pilots; the first that passes is the arm's setting for the study.

Importable without Isaac Lab; `apply` edits an env config object in place.
"""
BASE_TASK = "Sim2Gate-Velocity-Flat-Go2-P0-v5b"
TASK_PREFIX = "Sim2Gate-Velocity-Flat-Go2-P2"

ARMS = {
    "R-slip": {"cause": "reward", "symptom": "slip",
               "candidates": [{"id": "k0.1", "reward_scale": {"feet_slide": 0.1}},
                              {"id": "k0.03", "reward_scale": {"feet_slide": 0.03}}]},
    "Phi-slip": {"cause": "physics", "symptom": "slip",
                 "candidates": [{"id": "low", "static_friction": (0.2, 0.5)},
                                {"id": "high", "static_friction": (1.0, 1.2)}]},
    "R-jitter": {"cause": "reward", "symptom": "jitter",
                 "candidates": [{"id": "k0.1", "reward_scale": {"action_rate_l2": 0.1}},
                                {"id": "k0.03", "reward_scale": {"action_rate_l2": 0.03}}]},
    "Phi-jitter": {"cause": "physics", "symptom": "jitter",
                   "candidates": [{"id": "nodelay", "max_delay_steps": 0}]},
    "R-hop": {"cause": "reward", "symptom": "hop",
              "candidates": [{"id": "m10", "reward_scale": {"feet_air_time": 10.0}},
                             {"id": "m30", "reward_scale": {"feet_air_time": 30.0}}]},
}
ARM_ORDER = ("R-slip", "Phi-slip", "R-jitter", "Phi-jitter", "R-hop")
DYNAMIC_RATIO = 0.75                       # dynamic friction = 0.75 x static, as the task's own ratio


def candidate(arm, cid):
    for c in ARMS[arm]["candidates"]:
        if c["id"] == cid:
            return c
    raise KeyError(f"{arm}: no candidate {cid!r}")


def task_id(arm, cid):
    return f"{TASK_PREFIX}-{arm}-{cid}"


def all_tasks():
    return {task_id(a, c["id"]): (a, c["id"]) for a in ARM_ORDER for c in ARMS[a]["candidates"]}


def apply(cfg, arm, cid):
    """Edit a P0 v5b env config in place for this arm and candidate. Refuses anything that would remove a term."""
    c = candidate(arm, cid)
    for term, k in (c.get("reward_scale") or {}).items():
        t = getattr(cfg.rewards, term, None)
        if t is None:
            raise ValueError(f"{arm}: reward term {term!r} not in the base recipe")
        if not k or t.weight == 0:
            raise ValueError(f"{arm}: a phase 2 arm never sets a weight to zero")
        t.weight = t.weight * k
    if "static_friction" in c:
        lo, hi = c["static_friction"]
        pm = cfg.events.physics_material.params
        pm["static_friction_range"] = (lo, hi)
        pm["dynamic_friction_range"] = (lo * DYNAMIC_RATIO, hi * DYNAMIC_RATIO)
        pm["make_consistent"] = True
    if "max_delay_steps" in c:
        legs = cfg.scene.robot.actuators["base_legs"]
        if not hasattr(legs, "max_delay"):
            raise RuntimeError("P0 v5b's delayed DC motor was expected on base_legs")
        legs.min_delay = 0
        legs.max_delay = int(c["max_delay_steps"])
    return cfg


def symptom_present(symptom, metrics, hop_reference):
    """Part A section 3. metrics: a check 2 report's metrics. hop_reference: the controls' maxima of
    flight_share_low_speed and base_vz_rms_low_speed_mps."""
    from .layer1_limits import load_limit
    lim = load_limit()
    mx = lim["clean_maxima"]
    if symptom == "slip":
        return _gt(metrics.get("foot_slip_p95_mps"), mx["foot_slip_p95_mps"])
    if symptom == "jitter":
        return _gt(metrics.get("action_jitter_high_freq_power_share"), mx["action_jitter_high_freq_power_share"])
    if symptom == "hop":
        return (_gt(metrics.get("flight_share_low_speed"), hop_reference["flight_share_low_speed"])
                and _gt(metrics.get("base_vz_rms_low_speed_mps"), hop_reference["base_vz_rms_low_speed_mps"]))
    raise ValueError(symptom)


def _gt(v, ref):
    return isinstance(v, (int, float)) and isinstance(ref, (int, float)) and v > ref
