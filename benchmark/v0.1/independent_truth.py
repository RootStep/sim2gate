"""Independent behavioural-truth library for the Phase 2 tier-R development evaluation (spec section 5; gate G1).

Pure Python standard library. It deliberately imports nothing from Sim2Gate (no sim2gate.checks, no sim2gate.v2,
no frozen threshold files), so truth labels cannot reuse Sim2Gate's measurement code or limits. It works on raw
per-episode trajectories sampled at the policy (control) rate. A separate collector, to be written and run only
after G2 approval, produces them.

Differences from the Sim2Gate checks, by design:
  - Slip: stance-phase foot PATH LENGTH at the control rate, with touchdown/lift-off trimmed, reported per stance
    and as a ratio to base travel. Check 2 uses the 95th percentile of instantaneous horizontal foot speed at every
    physics update in contact.
  - Primary truth is CONSEQUENCE-based: an episode fails if the robot falls (own detector from base height and
    tilt, not the environment's termination flag) or loses velocity tracking.
  - Latency: an injectable buffer delays observations or actions by whole control steps. Check 1 sweeps actuator
    command delay in physics steps inside the motor model.
  - Manifestation is decided per policy against matched healthy controls under identical conditions (one-sided
    Fisher exact test plus a minimum rate difference), not by a fixed calibrated limit.

Every numeric rule is a parameter. DEFAULT_RULES are PROPOSALS only; the G2 execution manifest fixes the values
before any trajectory exists.
"""
import math

SCHEMA_VERSION = "independent-truth-episode-v1"

DEFAULT_RULES = {  # PROPOSED values; frozen only in the G2 manifest
    "contact_force_n": 10.0,        # foot in contact when vertical contact force exceeds this
    "min_stance_steps": 3,          # a stance phase must last at least this many control steps
    "stance_trim_steps": 1,         # control steps dropped at each end of a stance (touchdown / lift-off)
    "fall_height_m": 0.20,          # base height below this ...
    "fall_tilt_deg": 60.0,          # ... or base tilt above this ...
    "fall_persist_steps": 5,        # ... for this many consecutive control steps = fall
    "tracking_warmup_steps": 50,    # control steps ignored at the start of an episode
    "tracking_loss_mps": 0.25,      # episode fails if mean xy tracking error exceeds this
    "alpha": 0.01,                  # one-sided Fisher exact significance level
    "min_rate_difference": 0.10,    # and failure-rate difference vs controls of at least this
}
REQUIRED_RULES = tuple(DEFAULT_RULES)


class SchemaError(ValueError):
    pass


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_episode(ep):
    """Strict check of one raw episode record; returns its length T."""
    if not isinstance(ep, dict) or ep.get("schema") != SCHEMA_VERSION:
        raise SchemaError("not an independent-truth episode record")
    if not (_finite(ep.get("dt")) and ep["dt"] > 0):
        raise SchemaError("dt must be a positive number (seconds per control step)")
    T = len(ep.get("base_height_m") or [])
    if T == 0:
        raise SchemaError("empty episode")
    series = {"base_height_m": 1, "projected_gravity_b": 3, "base_lin_vel_b_xy": 2, "cmd_vel_xy": 2,
              "base_pos_w_xy": 2}
    for key, width in series.items():
        s = ep.get(key)
        if not isinstance(s, list) or len(s) != T:
            raise SchemaError(f"{key}: expected {T} samples")
        for v in s:
            vals = [v] if width == 1 else v
            if width > 1 and (not isinstance(v, list) or len(v) != width):
                raise SchemaError(f"{key}: each sample needs {width} values")
            if not all(_finite(x) for x in vals):
                raise SchemaError(f"{key}: non-finite value")
    feet = ep.get("feet")
    if not isinstance(feet, dict) or not feet:
        raise SchemaError("feet: expected a non-empty mapping")
    for name, f in feet.items():
        if not isinstance(f, dict) or len(f.get("contact_force_n") or []) != T or len(f.get("pos_w_xy") or []) != T:
            raise SchemaError(f"feet[{name}]: contact_force_n and pos_w_xy need {T} samples")
        if not all(_finite(x) for x in f["contact_force_n"]):
            raise SchemaError(f"feet[{name}]: non-finite contact force")
        if not all(isinstance(p, list) and len(p) == 2 and all(_finite(x) for x in p) for p in f["pos_w_xy"]):
            raise SchemaError(f"feet[{name}]: pos_w_xy samples need 2 finite values")
    return T


def _rules(rules):
    r = dict(rules or {})
    missing = [k for k in REQUIRED_RULES if k not in r]
    if missing:
        raise ValueError(f"rules missing: {', '.join(missing)}")
    return r


# ---- falls and tracking ----------------------------------------------------------------------------------------

def tilt_deg(gravity_b):
    """Angle between the base z axis and vertical, from gravity projected into the base frame."""
    gx, gy, gz = gravity_b
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    if n == 0:
        raise ValueError("zero gravity vector")
    return math.degrees(math.acos(max(-1.0, min(1.0, -gz / n))))


def detect_fall(ep, rules):
    """First control step of a sustained fall (low base or large tilt), or None."""
    r = _rules(rules)
    run = 0
    for t, (h, g) in enumerate(zip(ep["base_height_m"], ep["projected_gravity_b"])):
        bad = h < r["fall_height_m"] or tilt_deg(g) > r["fall_tilt_deg"]
        run = run + 1 if bad else 0
        if run >= r["fall_persist_steps"]:
            return t - run + 1
    return None


def tracking_error(ep, rules, end=None):
    """Mean xy velocity-tracking error after warm-up, up to `end` (exclusive); None if no samples remain."""
    r = _rules(rules)
    end = len(ep["base_height_m"]) if end is None else end
    errs = [math.hypot(c[0] - v[0], c[1] - v[1])
            for c, v in zip(ep["cmd_vel_xy"][r["tracking_warmup_steps"]:end],
                            ep["base_lin_vel_b_xy"][r["tracking_warmup_steps"]:end])]
    return sum(errs) / len(errs) if errs else None


def episode_outcome(ep, rules):
    """Primary consequence-based outcome of one episode."""
    validate_episode(ep)
    r = _rules(rules)
    fall_at = detect_fall(ep, r)
    err = tracking_error(ep, r, end=fall_at)
    tracking_loss = err is not None and err > r["tracking_loss_mps"]
    return {"fell": fall_at is not None, "fall_step": fall_at, "tracking_error_mps": err,
            "tracking_loss": tracking_loss, "failure": fall_at is not None or tracking_loss}


# ---- stance-phase slip (secondary magnitude) -------------------------------------------------------------------

def stance_phases(contact_force, rules):
    """(start, end_exclusive) control-step ranges of sustained contact, trimmed at both ends."""
    r = _rules(rules)
    out, start = [], None
    for t, f in enumerate(list(contact_force) + [0.0]):
        on = f > r["contact_force_n"]
        if on and start is None:
            start = t
        elif not on and start is not None:
            if t - start >= r["min_stance_steps"]:
                a, b = start + r["stance_trim_steps"], t - r["stance_trim_steps"]
                if b - a >= 2:
                    out.append((a, b))
            start = None
    return out


def _path(points):
    return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(points, points[1:]))


def _quantile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stance_slip(ep, rules, end=None):
    """Per-stance foot path length (m) and its ratio to base travel, up to `end` (exclusive)."""
    validate_episode(ep)
    r = _rules(rules)
    end = len(ep["base_height_m"]) if end is None else end
    per_stance = []
    for f in ep["feet"].values():
        for a, b in stance_phases(f["contact_force_n"][:end], r):
            per_stance.append(_path(f["pos_w_xy"][a:b]))
    travel = _path(ep["base_pos_w_xy"][:end])
    return {"stances": len(per_stance), "median_m": _quantile(per_stance, 0.5), "p90_m": _quantile(per_stance, 0.9),
            "total_m": sum(per_stance), "base_travel_m": travel,
            "slip_ratio": (sum(per_stance) / travel) if travel > 0 else None}


# ---- matched-control manifestation decision --------------------------------------------------------------------

def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_one_sided_greater(a, b, c, d):
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] with fixed margins (hypergeometric): policy failures a of a+b,
    control failures c of c+d. Small p means the policy fails more often than controls."""
    for x in (a, b, c, d):
        if not isinstance(x, int) or x < 0:
            raise ValueError("counts must be non-negative integers")
    n1, k, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, k - (c + d)), min(k, n1)
    denom = _log_comb(n, k)
    return min(1.0, sum(math.exp(_log_comb(n1, x) + _log_comb(n - n1, k - x) - denom) for x in range(max(a, lo), hi + 1)))


def manifestation(policy_failures, policy_n, control_failures, control_n, rules):
    """Does a policy fail more than matched healthy controls under identical conditions?"""
    r = _rules(rules)
    if policy_n <= 0 or control_n <= 0 or not (0 <= policy_failures <= policy_n) or not (0 <= control_failures <= control_n):
        raise ValueError("invalid counts")
    p = fisher_one_sided_greater(policy_failures, policy_n - policy_failures, control_failures, control_n - control_failures)
    diff = policy_failures / policy_n - control_failures / control_n
    return {"p_value": p, "rate_difference": diff, "policy_rate": policy_failures / policy_n,
            "control_rate": control_failures / control_n,
            "manifested": p < r["alpha"] and diff >= r["min_rate_difference"]}


# ---- latency injection -----------------------------------------------------------------------------------------

class LatencyBuffer:
    """Delays a stream (observations or actions) by `steps` whole control steps. Before enough history exists it
    returns the first value received, so a delayed policy starts from a defined state."""

    def __init__(self, steps):
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        self.steps, self._hist = steps, []

    def reset(self):
        self._hist = []

    def __call__(self, value):
        self._hist.append(value)
        if len(self._hist) > self.steps + 1:
            self._hist.pop(0)
        return self._hist[0]
