"""Check 5, sim-to-sim transfer: parity conditions and scoring (no Isaac Lab needed).

MVP plan, check 5: the same policy runs under PhysX (the training engine) and Newton (MuJoCo-Warp solver), closed
loop, fixed command set, 64 episodes per engine. Flagged if Newton's success rate is below 80% or the ratio of mean
velocity-tracking error, Newton / PhysX, exceeds 2. Parity conditions, all required, else the check is invalid
(exit code 3) and not scored:

1. Same configuration: identical task, policy checkpoint, command set, control rate, seeds, environment count,
   nominal profile and (physics-independent) environment config; only the physics preset differs.
2. Nominal sanity: the policy reaches 80% episode success on PhysX; otherwise "policy not ready".
3. Engine settings recorded and not tuned: the Newton run records its solver settings and versions, and neither run
   carries any physics override beyond the engine choice and the fixed actuator-path setting below.

Both runs use Isaac Lab's own actuator model (env.sim.use_newton_actuators=false), so the DC-motor code is identical
and only the physics engine differs; the Go2 task otherwise runs its motors natively in Newton.
"""
from .robustness import CLIFF_SUCCESS, check_compatible, success_rate

NEWTON_SUCCESS_MIN = 0.8
ERROR_RATIO_MAX = 2.0
PHYSX = "physics=isaacsim_physx"
NEWTON = "physics=newton_mjwarp"
# the only physics-related overrides allowed, identical for every policy
FIXED_OVERRIDES = ("env.sim.use_newton_actuators=false",)
SAME = ("task", "checkpoint_sha256", "command_set_sha256", "step_dt", "decimation", "sim_dt", "seed", "num_envs",
        "episode_steps", "profile", "env_cfg_sha256")


def _physics_overrides(raw):
    return sorted(o for o in raw.get("overrides", []) if o.split("=")[0].startswith(("physics", "env.sim")))


def parity(physx, newton):
    """List of parity failures (empty if the runs are comparable)."""
    bad = []
    for k in SAME:
        a, b = physx.get(k), newton.get(k)
        if a is None or b is None:
            bad.append(f"same configuration: {k} not recorded")
        elif a != b:
            bad.append(f"same configuration: {k} differs (PhysX {a!r}, Newton {b!r})")
    allowed_p, allowed_n = sorted([PHYSX, *FIXED_OVERRIDES]), sorted([NEWTON, *FIXED_OVERRIDES])
    if _physics_overrides(physx) != allowed_p:
        bad.append(f"engine settings: the PhysX run's physics overrides are {_physics_overrides(physx)}, "
                   f"expected {allowed_p}")
    if _physics_overrides(newton) != allowed_n:
        bad.append(f"engine settings: the Newton run's physics overrides are {_physics_overrides(newton)}, "
                   f"expected {allowed_n}")
    for label, r, want in (("PhysX", physx, "physx"), ("Newton", newton, "newton")):
        eng = (r.get("physics") or {}).get("manager") or ""
        if want not in eng.lower():
            bad.append(f"engine settings: the {label} run's physics manager is {eng!r}")
    nphys = newton.get("physics") or {}
    if not nphys.get("solver_cfg"):
        bad.append("engine settings: the Newton run did not record its solver settings")
    if not nphys.get("versions"):
        bad.append("engine settings: the Newton run did not record its engine versions")
    if (newton.get("conditions") or {}).get("varied"):
        bad.append("same configuration: the Newton run varied physical parameters")
    if (physx.get("conditions") or {}).get("varied"):
        bad.append("same configuration: the PhysX run varied physical parameters")
    return bad


def _mean_error(raw):
    e = [x["mean_error_mps"] for x in raw["episodes"] if x.get("mean_error_mps") is not None]
    return sum(e) / len(e) if e else None


def score(physx, newton, success):
    thr = success["threshold_mps"]
    reasons = parity(physx, newton)
    reasons += check_compatible(physx, success, "PhysX run") + check_compatible(newton, success, "Newton run")
    sp, sn = success_rate(physx["episodes"], thr), success_rate(newton["episodes"], thr)
    ep, en = _mean_error(physx), _mean_error(newton)
    ratio = (en / ep) if (ep and en is not None) else None
    if sp is None or sp < CLIFF_SUCCESS:
        reasons.append(f"policy not ready: PhysX success rate {sp} is below {CLIFF_SUCCESS}")
    out = {"check": 5, "success_threshold_mps": thr,
           "metrics": {"physx_success_rate": sp, "newton_success_rate": sn, "physx_mean_error_mps": ep,
                       "newton_mean_error_mps": en, "error_ratio": ratio,
                       "physx_falls": sum(bool(x["fell"]) for x in physx["episodes"]),
                       "newton_falls": sum(bool(x["fell"]) for x in newton["episodes"])},
           "rule": {"newton_success_min": NEWTON_SUCCESS_MIN, "error_ratio_max": ERROR_RATIO_MAX},
           "parity": "passed" if not reasons else "failed", "reasons": reasons, "risks": []}
    if not reasons:
        if sn is not None and sn < NEWTON_SUCCESS_MIN:
            out["risks"].append({"check": 5, "metric": "newton_success_rate", "value": sn,
                                 "threshold": NEWTON_SUCCESS_MIN, "margin": sn - NEWTON_SUCCESS_MIN})
        if ratio is not None and ratio > ERROR_RATIO_MAX:
            out["risks"].append({"check": 5, "metric": "error_ratio", "value": ratio, "threshold": ERROR_RATIO_MAX,
                                 "margin": ERROR_RATIO_MAX - ratio})
        for r in out["risks"]:
            r.update({"likely_cause": "the policy depends on PhysX contact behaviour that Newton does not reproduce",
                      "fix": "review contact parameters and randomize them in training, then retrain"})
    out["status"] = "invalid" if reasons else "scored"
    out["flagged"] = None if reasons else bool(out["risks"])
    return out
