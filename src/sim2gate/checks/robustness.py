"""Check 1, robustness envelope: the sweep grid and its scoring (no Isaac Lab needed).

MVP plan, check 1: sweep one parameter at a time, all others at nominal, 10 evenly spaced values from nominal to the
edge of the plausible real range plus 50%, 64 episodes per value. The cliff is the first value where the success
rate drops below 80%; a parameter is flagged when its cliff lies inside its plausible real range.

Reading of the plan used here (recorded in every report as `grid_definition`):
- "Plus 50%" extends the sweep by half the nominal-to-edge distance: the last value is nominal + 1.5 x (edge -
  nominal). The plausible edge is then exactly the 7th of the 10 values.
- Each parameter is swept in the direction that moves away from training conditions towards harder conditions:
  friction down from the task's 0.8 towards the plausible 0.4 (the plausible range's upper end, 1.2, makes walking
  easier, not harder), added mass up, motor strength down, latency up, noise up.
- Friction is the static coefficient of the robot's contact material; dynamic friction keeps the task's ratio
  (0.6 / 0.8). The ground's material is 1.0 with multiply combining, so these are the effective coefficients.
- All 9 non-nominal values of every parameter share one nominal group (index 0) of 64 episodes.

Episode success (plan, "Scope and definitions"): the 20 s episode ends without a fall, and its mean linear-velocity
tracking error is at most 1.5 x the clean baseline's mean error under nominal conditions. The clean baseline's
error is measured by this check's own nominal condition on the calibration baselines, P0 seeds 1-3
(`success_reference`), so success is judged under the same conditions it was calibrated in.
"""
import hashlib
import json
import math

GRID_VERSION = 1
N_VALUES = 10
EXTEND = 1.5               # last value = nominal + EXTEND x (edge - nominal)
EPISODES_PER_VALUE = 64
CLIFF_SUCCESS = 0.8        # cliff: first value whose success rate is below this
SUCCESS_ERROR_FACTOR = 1.5
# A calibration baseline may fall in at most this many of its nominal episodes: the same tolerance as the C4
# acceptance criteria (sim2gate.mvp.baselines BASELINE_CRITERIA["max_falls"]). Fallen episodes are left out of the
# reference (plan change C8).
MAX_NOMINAL_FALLS = 1

# name -> nominal (training condition), edge (plausible real edge in the swept direction), plausible range
PARAMS = {
    "ground_friction": {"nominal": 0.8, "edge": 0.4, "plausible": [0.4, 1.2], "unit": "static coefficient",
                        "fix": "randomize ground friction over the plausible range (0.4-1.2) in training, then retrain"},
    "added_base_mass_kg": {"nominal": 0.0, "edge": 3.0, "plausible": [0.0, 3.0], "unit": "kg",
                           "fix": "randomize added base mass over 0-3 kg in training, then retrain"},
    "motor_strength_scale": {"nominal": 1.0, "edge": 0.85, "plausible": [0.85, 1.0], "unit": "x nominal torque",
                             "fix": "randomize motor strength over 0.85-1.0 in training, then retrain"},
    "action_latency_ms": {"nominal": 0.0, "edge": 30.0, "plausible": [0.0, 30.0], "unit": "ms",
                          "fix": "randomize action latency over 0-30 ms in training, then retrain"},
    "observation_noise_scale": {"nominal": 1.0, "edge": 2.0, "plausible": [1.0, 2.0], "unit": "x task noise",
                                "fix": "train with observation noise up to 2x the task default, then retrain"},
}
DYNAMIC_FRICTION_RATIO = 0.6 / 0.8


def values(name):
    p = PARAMS[name]
    end = p["nominal"] + EXTEND * (p["edge"] - p["nominal"])
    return [round(p["nominal"] + (end - p["nominal"]) * i / (N_VALUES - 1), 10) for i in range(N_VALUES)]


def nominal_condition():
    return {k: p["nominal"] for k, p in PARAMS.items()}


def conditions():
    """The sweep's groups, in env order: group 0 is nominal (shared), then 9 values per parameter in PARAMS order.
    Each: {"group", "param" (None for nominal), "index" (0..9), "value", "condition": full parameter dict}."""
    out = [{"group": 0, "param": None, "index": 0, "value": None, "condition": nominal_condition()}]
    for name in PARAMS:
        for i, v in enumerate(values(name)[1:], start=1):
            c = nominal_condition()
            c[name] = v
            out.append({"group": len(out), "param": name, "index": i, "value": v, "condition": c})
    return out


def grid_definition():
    d = {"version": GRID_VERSION, "n_values": N_VALUES, "extend": EXTEND, "episodes_per_value": EPISODES_PER_VALUE,
         "cliff_success": CLIFF_SUCCESS, "params": {k: {**p, "values": values(k)} for k, p in PARAMS.items()},
         "dynamic_friction_ratio": DYNAMIC_FRICTION_RATIO,
         "reading": "last value = nominal + 1.5 x (edge - nominal); swept from nominal towards harder conditions; "
                    "one shared nominal group"}
    d["sha256"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d


def layout(groups, episodes):
    """env index -> group index, `episodes` consecutive envs per group."""
    return [g["group"] for g in groups for _ in range(episodes)]


# ---- episode success -------------------------------------------------------------------------------------------

def episode_ok(ep, threshold):
    """One episode record {"fell": bool, "mean_error_mps": float|None, ...} -> success (bool)."""
    if ep.get("fell"):
        return False
    e = ep.get("mean_error_mps")
    return e is not None and math.isfinite(e) and e <= threshold + 1e-12


def success_rate(episodes, threshold):
    eps = list(episodes)
    return (sum(episode_ok(e, threshold) for e in eps) / len(eps)) if eps else None


class SuccessError(ValueError):
    pass


def success_reference(raws, factor=SUCCESS_ERROR_FACTOR, max_falls=MAX_NOMINAL_FALLS):
    """The episode-success error threshold from the calibration baselines' nominal episodes.

    raws: list of (label, raw report) from P0 seeds 1-3, run with the same nominal profile and command set.
    reference = mean, over all their nominal episodes that did not fall, of each episode's mean tracking error;
    threshold = factor x reference. A baseline with more than max_falls nominal falls is refused (the C4 tolerance);
    fallen episodes are left out of the reference and listed (plan change C8)."""
    if len(raws) < 1:
        raise SuccessError("no calibration reports")
    keys = ("profile", "command_set_sha256", "episode_steps")
    first = raws[0][1]
    errs, sources = [], []
    for label, r in raws:
        for k in keys:
            if r.get(k) != first.get(k):
                raise SuccessError(f"{label}: {k} differs from {raws[0][0]} ({r.get(k)!r} vs {first.get(k)!r})")
        nom = [e for e in r["episodes"] if e["group"] == 0]
        if not nom:
            raise SuccessError(f"{label}: no nominal episodes")
        fell = [e for e in nom if e["fell"]]
        if len(fell) > max_falls:
            raise SuccessError(f"{label}: {len(fell)} nominal episode(s) fell; a calibration baseline may fall in at "
                               f"most {max_falls}")
        kept = [e for e in nom if not e["fell"]]
        errs += [e["mean_error_mps"] for e in kept]
        sources.append({"label": label, "checkpoint_sha256": (r.get("checkpoint") or {}).get("sha256"),
                        "nominal_episodes": len(nom), "used_episodes": len(kept),
                        "excluded_fallen": [{"env": e.get("env"), "steps": e.get("steps"),
                                             "mean_error_mps": e["mean_error_mps"]} for e in fell],
                        "mean_error_mps": sum(e["mean_error_mps"] for e in kept) / len(kept)})
    ref = sum(errs) / len(errs)
    return {"version": 2, "rule": f"episode success: no fall and mean tracking error <= {factor} x reference",
            "reference_rule": f"mean over nominal episodes without a fall; at most {max_falls} fall per baseline (C8)",
            "reference_error_mps": ref, "factor": factor, "threshold_mps": factor * ref,
            "profile": first.get("profile"), "command_set_sha256": first.get("command_set_sha256"),
            "episode_steps": first.get("episode_steps"), "sources": sources}


def load_success(path):
    doc = json.load(open(path))
    for k in ("threshold_mps", "profile", "command_set_sha256", "episode_steps"):
        if k not in doc:
            raise SuccessError(f"{path}: missing {k}")
    return doc


def check_compatible(raw, success, label="run"):
    """Problems that make `success` inapplicable to `raw` (empty list if none)."""
    bad = []
    for k in ("profile", "command_set_sha256", "episode_steps"):
        if raw.get(k) != success.get(k):
            bad.append(f"{label}: {k} {raw.get(k)!r} differs from the success calibration's {success.get(k)!r}")
    return bad


# ---- check 1 scoring --------------------------------------------------------------------------------------------

def _inside(v, rng):
    lo, hi = min(rng), max(rng)
    return lo - 1e-9 <= v <= hi + 1e-9


def score(raw, success):
    """Check 1 result from a sweep's raw episodes and the success calibration.

    Returns {"status": "scored"|"invalid", "flagged": bool|None, "nominal_success_rate", "params": {name: {...}},
    "risks": [...], "reasons": [...]}. invalid (the plan's exit code 3) when the success calibration does not apply
    or the policy does not reach 80% success at nominal ("policy not ready")."""
    reasons = check_compatible(raw, success)
    grid = raw.get("grid") or {}
    if grid.get("sha256") != grid_definition()["sha256"]:
        reasons.append("the sweep grid differs from this version's grid definition")
    thr = success["threshold_mps"]
    by_group = {}
    for e in raw["episodes"]:
        by_group.setdefault(e["group"], []).append(e)
    groups = {g["group"]: g for g in raw["groups"]}
    rates = {g: success_rate(eps, thr) for g, eps in by_group.items()}
    nominal = rates.get(0)
    out = {"check": 1, "success_threshold_mps": thr, "nominal_success_rate": nominal,
           "nominal_episodes": len(by_group.get(0, [])), "params": {}, "risks": [], "reasons": reasons}
    if nominal is None:
        reasons.append("no nominal episodes")
    elif nominal < CLIFF_SUCCESS:
        reasons.append(f"policy not ready: nominal success rate {nominal:.2f} is below {CLIFF_SUCCESS}")
    for name, p in PARAMS.items():
        series = [{"index": 0, "value": p["nominal"], "success_rate": nominal, "episodes": len(by_group.get(0, []))}]
        for g in sorted((g for g in groups.values() if g["param"] == name), key=lambda g: g["index"]):
            series.append({"index": g["index"], "value": g["value"], "success_rate": rates.get(g["group"]),
                           "episodes": len(by_group.get(g["group"], []))})
        cliff = next((s for s in series if s["success_rate"] is not None and s["success_rate"] < CLIFF_SUCCESS), None)
        flagged = cliff is not None and _inside(cliff["value"], p["plausible"])
        out["params"][name] = {"series": series, "cliff": cliff["value"] if cliff else None,
                               "cliff_index": cliff["index"] if cliff else None, "plausible": p["plausible"],
                               "flagged": flagged, "unit": p["unit"],
                               "training_range": (raw.get("training_ranges") or {}).get(name)}
        if flagged:
            edge = p["edge"]
            out["risks"].append({
                "check": 1, "metric": f"{name} cliff", "value": cliff["value"], "threshold": edge,
                "margin": round(abs(edge - p["nominal"]) - abs(cliff["value"] - p["nominal"]), 10),
                "evidence": series, "likely_cause": "the policy was not trained across this parameter's real range",
                "fix": p["fix"], "training_range": out["params"][name]["training_range"]})
    out["status"] = "invalid" if reasons else "scored"
    out["flagged"] = None if reasons else bool(out["risks"])
    return out
