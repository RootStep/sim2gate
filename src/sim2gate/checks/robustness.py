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

# v2 operating envelope (v2 plan, "Layer 1: operating envelope"): payload to 8 kg (the Go2's rated payload), friction
# down to 0.35, motor strength down to 0.8; latency and noise as v1. Same grid rule (10 values, plausible edge + 50%).
PARAMS_V2 = {
    "ground_friction": {"nominal": 0.8, "edge": 0.35, "plausible": [0.35, 1.2], "unit": "static coefficient",
                        "fix": "randomize ground friction over 0.35-1.2 in training, then retrain"},
    "added_base_mass_kg": {"nominal": 0.0, "edge": 8.0, "plausible": [0.0, 8.0], "unit": "kg",
                           "fix": "randomize added payload over 0-8 kg in training, then retrain"},
    "motor_strength_scale": {"nominal": 1.0, "edge": 0.8, "plausible": [0.8, 1.0], "unit": "x nominal torque",
                             "fix": "randomize motor strength over 0.8-1.0 in training, then retrain"},
    "action_latency_ms": dict(PARAMS["action_latency_ms"]),
    "observation_noise_scale": dict(PARAMS["observation_noise_scale"]),
}
ENVELOPES = {"v1": PARAMS, "v2": PARAMS_V2}


def params(envelope="v1"):
    if envelope not in ENVELOPES:
        raise ValueError(f"unknown envelope {envelope!r} (known: {sorted(ENVELOPES)})")
    return ENVELOPES[envelope]


def values(name, envelope="v1"):
    p = params(envelope)[name]
    end = p["nominal"] + EXTEND * (p["edge"] - p["nominal"])
    return [round(p["nominal"] + (end - p["nominal"]) * i / (N_VALUES - 1), 10) for i in range(N_VALUES)]


def nominal_condition(envelope="v1"):
    return {k: p["nominal"] for k, p in params(envelope).items()}


def conditions(envelope="v1"):
    """The sweep's groups, in env order: group 0 is nominal (shared), then 9 values per parameter in PARAMS order.
    Each: {"group", "param" (None for nominal), "index" (0..9), "value", "condition": full parameter dict}."""
    out = [{"group": 0, "param": None, "index": 0, "value": None, "condition": nominal_condition(envelope)}]
    for name in params(envelope):
        for i, v in enumerate(values(name, envelope)[1:], start=1):
            c = nominal_condition(envelope)
            c[name] = v
            out.append({"group": len(out), "param": name, "index": i, "value": v, "condition": c})
    return out


def grid_definition(envelope="v1"):
    d = {"version": GRID_VERSION, "n_values": N_VALUES, "extend": EXTEND, "episodes_per_value": EPISODES_PER_VALUE,
         "cliff_success": CLIFF_SUCCESS,
         "params": {k: {**p, "values": values(k, envelope)} for k, p in params(envelope).items()},
         "dynamic_friction_ratio": DYNAMIC_FRICTION_RATIO,
         "reading": "last value = nominal + 1.5 x (edge - nominal); swept from nominal towards harder conditions; "
                    "one shared nominal group"}
    if envelope != "v1":                      # v1's definition (and its hash) stays exactly as frozen
        d["envelope"] = envelope
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
    envelope = grid.get("envelope", "v1")
    if envelope not in ENVELOPES or grid.get("sha256") != grid_definition(envelope)["sha256"]:
        reasons.append("the sweep grid differs from this version's grid definition")
        envelope = "v1"
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
    out["envelope"] = envelope
    for name, p in params(envelope).items():
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


# ---- joint sweep (report only; MVP plan, check 1 "Also run") ------------------------------------------------------
# All five parameters vary together. Each episode draws each parameter independently and uniformly from its 10 sweep
# values (the plan asks for a uniform draw, which Arena's sensitivity analyzer assumes; the grid values keep latency
# on whole physics steps and friction to a few material settings). Same number of episodes as the one-at-a-time
# sweep. Never a flag in the MVP: the verdict ignores it.
JOINT_VERSION = 1
JOINT_EPISODES = 46 * EPISODES_PER_VALUE     # 2,944, the one-at-a-time sweep's size
JOINT_SEED = 20261016
HARD_HALF = 0.5                              # "harder half of the realistic range": severity above this, up to 1


def joint_conditions(n=JOINT_EPISODES, seed=JOINT_SEED):
    import random
    rng = random.Random(seed)
    vals = {k: values(k) for k in PARAMS}
    return [{"group": i, "param": "joint", "index": None, "value": None,
             "condition": {k: rng.choice(vals[k]) for k in PARAMS}} for i in range(n)]


def joint_definition(n=JOINT_EPISODES, seed=JOINT_SEED):
    conds = [g["condition"] for g in joint_conditions(n, seed)]
    d = {"version": JOINT_VERSION, "episodes": n, "seed": seed, "grid_sha256": grid_definition()["sha256"],
         "sampling": "each parameter independently and uniformly from its 10 sweep values",
         "conditions_sha256": hashlib.sha256(json.dumps(conds, sort_keys=True).encode()).hexdigest()}
    d["sha256"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d


def severity(name, v):
    """0 at nominal, 1 at the plausible edge, 1.5 at the end of the sweep."""
    p = PARAMS[name]
    return (v - p["nominal"]) / (p["edge"] - p["nominal"])


def _rate(eps):
    return (sum(e["ok"] for e in eps) / len(eps)) if eps else None


def joint_report(raw, success):
    """Report-only summary of a joint sweep: success when every condition is realistic at once, per-value marginal
    success rates, and pairs of conditions that fail more together than either does alone. Plus the per-episode
    table in the form Arena's sensitivity analyzer reads (factors, theta, x), to run its analysis after the port."""
    reasons = check_compatible(raw, success)
    if raw.get("mode") != "joint":
        reasons.append(f"not a joint sweep (mode {raw.get('mode')!r})")
    if (raw.get("grid") or {}).get("sha256") != joint_definition()["sha256"]:
        reasons.append("the joint sweep differs from this version's definition")
    out = {"check": "1-joint", "report_only": True, "flagged": None, "reasons": reasons}
    if reasons:
        out["status"] = "invalid"
        return out
    thr = success["threshold_mps"]
    cond = {g["group"]: g["condition"] for g in raw["groups"]}
    eps = []
    for e in raw["episodes"]:
        c = cond[e["group"]]
        eps.append({"ok": episode_ok(e, thr), "sev": {k: severity(k, c[k]) for k in PARAMS}, "c": c})
    realistic = [e for e in eps if all(s <= 1 + 1e-9 for s in e["sev"].values())]
    beyond = [e for e in eps if any(s > 1 + 1e-9 for s in e["sev"].values())]
    out.update(status="scored", episodes=len(eps), success_rate=_rate(eps),
               all_realistic={"episodes": len(realistic), "success_rate": _rate(realistic),
                              "meaning": "every condition inside its plausible real range at once"},
               any_beyond={"episodes": len(beyond), "success_rate": _rate(beyond)})
    out["marginal"] = {k: [{"value": v, "episodes": len(sel), "success_rate": _rate(sel)}
                           for v in values(k) for sel in [[e for e in eps if abs(e["c"][k] - v) < 1e-9]]]
                       for k in PARAMS}
    # pairs, inside the realistic box: both in their harder half vs each alone (the other in its easier half)
    names = list(PARAMS)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            hard = lambda e, k: HARD_HALF < e["sev"][k] <= 1 + 1e-9  # noqa: E731
            easy = lambda e, k: e["sev"][k] <= HARD_HALF  # noqa: E731
            both = [e for e in realistic if hard(e, a) and hard(e, b)]
            only_a = [e for e in realistic if hard(e, a) and easy(e, b)]
            only_b = [e for e in realistic if easy(e, a) and hard(e, b)]
            neither = [e for e in realistic if easy(e, a) and easy(e, b)]
            r = {"pair": [a, b], "both_hard": _rate(both), "a_hard_only": _rate(only_a), "b_hard_only": _rate(only_b),
                 "neither": _rate(neither), "episodes": [len(both), len(only_a), len(only_b), len(neither)]}
            if None not in (r["both_hard"], r["a_hard_only"], r["b_hard_only"], r["neither"]) and r["neither"] > 0:
                # independence: failures from each hardening multiply as success ratios
                expected = r["a_hard_only"] * r["b_hard_only"] / r["neither"]
                r["expected_if_independent"] = expected
                r["interaction"] = r["both_hard"] - expected      # negative: worse together than apart
            pairs.append(r)
    out["pairs"] = sorted(pairs, key=lambda r: r.get("interaction", 0.0))
    out["arena_dataset"] = {
        "factors": [{"name": k, "type": "continuous", "range": [min(values(k)), max(values(k))]} for k in PARAMS],
        "outcome_names": ["success"],
        "theta": [[e["c"][k] for k in PARAMS] for e in eps], "x": [[1.0 if e["ok"] else 0.0] for e in eps],
        "note": "isaaclab_arena.analysis.sensitivity.SensitivityDataset(factors, theta, x); continuous factors"}
    return out
