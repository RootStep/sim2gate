"""v2 Layer 1 limits from the 20 clean P0 v5b robots (docs/v2/LAYER1_LIMITS_PROTOCOL.md, frozen Oct 22, 2026).

Reads the measurement files extracted from the phase 1 bundle (phase1-clean.tgz) together with its MANIFEST.json,
checks every file against the manifest, and writes once:

    frozen_success_p0v5b.json   the v2 episode-success threshold (protocol section 2)
    check2_limit_v2.json        the combined check 2 limit (section 3)
    layer1_clean_report.json    checks 1, 2 and 5 on the clean robots, the stop rule and leave-one-out (sections 4-6)

    python -m sim2gate.v2.layer1_limits --root <folder holding phase1_clean/> --out <new folder>

Nothing here is tuned: the rules, metrics and floors are the frozen protocol's. The output folder must not exist yet.
"""
import argparse
import hashlib
import json
import math
import os
import sys

PROTOCOL = "docs/v2/LAYER1_LIMITS_PROTOCOL.md"
PROTOCOL_SHA256 = "3b72a48590c337160832c1862926e5bdd7b9984d38f5ff4d90ca964671498a18"
BUNDLE_SHA256 = "36bb9b75caf2ab7d37482bf7dc79c5ddd48cacfd2bde3f3aac834ef5c23b712a"
BUNDLE_DIR = "phase1_clean"
CALIBRATION_SEEDS = (101, 102, 103, 104, 105, 107, 108, 109, 110, 111,
                     113, 114, 115, 116, 117, 118, 119, 120, 121, 122)
NAME = "p0v5b_s{}"
FACTOR = 1.5                                       # section 2: v1's rule
METRICS = ("foot_slip_p95_mps", "ground_penetration_m", "actuator_limit_step_share",
           "action_jitter_high_freq_power_share")  # section 3; contact_force_p99_bw reported only
REPORTED_ONLY = ("contact_force_p99_bw",)
FLOORS = {"foot_slip_p95_mps": 0.001, "ground_penetration_m": 0.0001, "actuator_limit_step_share": 0.0001,
          "action_jitter_high_freq_power_share": 0.0001}
STOP_ABOVE = 3                                     # section 5: stop if more than 3 of 20 get a no-go
MEASURES = ("nominal", "sweep", "check2", "newton")


class LimitsError(ValueError):
    pass


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k, n, z=1.959963984540054):
    """95% Wilson interval for k of n."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---- loading -------------------------------------------------------------------------------------------------------

def load(root, seeds=CALIBRATION_SEEDS):
    """{name: {"nominal": raw, "sweep": raw, "check2": report, "newton": raw, "state": robot record}} after checking
    every file against MANIFEST.json and each measurement's checkpoint against the run state."""
    base = os.path.join(root, BUNDLE_DIR)
    mp = os.path.join(base, "MANIFEST.json")
    if not os.path.exists(mp):
        raise LimitsError(f"{mp} not found: extract MANIFEST.json and the measure/*.json files of the bundle")
    man = json.load(open(mp))
    files, robots = man.get("files") or {}, (man.get("state") or {}).get("robots") or {}
    calib = set((man.get("state") or {}).get("calibration_set") or [])
    out = {}
    for seed in seeds:
        name = NAME.format(seed)
        r = robots.get(name)
        if not r or not r.get("finished") or r.get("problems"):
            raise LimitsError(f"{name}: not a finished robot in the run state")
        if calib and name not in calib:
            raise LimitsError(f"{name}: not in the run's calibration set")
        nom = (r.get("summary") or {}).get("nominal") or {}
        if not nom.get("qualifies"):
            raise LimitsError(f"{name}: did not qualify")
        ck = (r.get("checkpoint") or {}).get("sha256")
        rec = {"state": r}
        for m in MEASURES:
            arc = f"robots/{name}/measure/{m}.json"
            path = os.path.join(base, arc)
            if arc not in files:
                raise LimitsError(f"{arc}: not in the bundle manifest")
            if not os.path.exists(path):
                raise LimitsError(f"{path}: missing (extract robots/*/measure/*.json)")
            if _sha(path) != files[arc]:
                raise LimitsError(f"{arc}: SHA-256 differs from the bundle manifest")
            doc = json.load(open(path))
            got = doc.get("checkpoint_sha256") or (doc.get("checkpoint") or {}).get("sha256")
            if ck and got and got != ck:
                raise LimitsError(f"{arc}: measured checkpoint {got[:12]} is not the robot's {ck[:12]}")
            rec[m] = doc
        out[name] = rec
    if len(out) != len(seeds):
        raise LimitsError("calibration set incomplete")
    return out


# ---- section 2: success threshold ---------------------------------------------------------------------------------

def success_threshold(nominals, factor=FACTOR):
    """nominals: {name: nominal raw}. Reference = mean over robots of each robot's mean tracking error over its
    nominal episodes without a fall; threshold = factor x reference. Same document shape as v1's success file."""
    keys = ("profile", "command_set_sha256", "episode_steps")
    names = sorted(nominals)
    first = nominals[names[0]]
    per, sources = [], []
    for n in names:
        r = nominals[n]
        for k in keys:
            if r.get(k) != first.get(k):
                raise LimitsError(f"{n}: {k} differs ({r.get(k)!r} vs {first.get(k)!r})")
        eps = [e for e in r["episodes"] if e.get("group", 0) == 0]
        kept = [e for e in eps if not e["fell"] and _fin(e.get("mean_error_mps"))]
        if not kept:
            raise LimitsError(f"{n}: no nominal episode without a fall")
        m = sum(e["mean_error_mps"] for e in kept) / len(kept)
        per.append(m)
        sources.append({"label": n, "checkpoint_sha256": r.get("checkpoint_sha256")
                        or (r.get("checkpoint") or {}).get("sha256"),
                        "nominal_episodes": len(eps), "used_episodes": len(kept),
                        "falls": sum(bool(e["fell"]) for e in eps), "mean_error_mps": m})
    ref = sum(per) / len(per)
    return {"version": 3, "protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256,
            "rule": f"episode success: no fall and mean tracking error <= {factor} x reference",
            "reference_rule": "mean over robots of each robot's mean error over nominal episodes without a fall "
                              "(each robot counts equally; fallen episodes left out and counted)",
            "reference_error_mps": ref, "factor": factor, "threshold_mps": factor * ref,
            "profile": first.get("profile"), "command_set_sha256": first.get("command_set_sha256"),
            "episode_steps": first.get("episode_steps"), "robots": len(names), "sources": sources}


# ---- section 3: combined check 2 limit ----------------------------------------------------------------------------

def _ratios(x, others_max):
    return {m: x[m] / max(others_max[m], FLOORS[m]) for m in METRICS}


def _floor_used(others_max):
    return [m for m in METRICS if others_max[m] < FLOORS[m]]


def clean_scores(values):
    """values: {name: {metric: value}}. Each robot's score against the highest value of the others."""
    out = {}
    for n, x in values.items():
        om = {m: max(v[m] for k, v in values.items() if k != n) for m in METRICS}
        rat = _ratios(x, om)
        drv = max(rat, key=lambda m: rat[m])
        out[n] = {"score": rat[drv], "driving_metric": drv, "ratios": rat, "floors_used": _floor_used(om)}
    return out


def combined_limit(reports):
    """reports: {name: check 2 report}. Returns the limit document (applicability fields as v1's thresholds file)."""
    from ..checks import calibrate
    raws = [(n, json.dumps(reports[n], sort_keys=True).encode()) for n in sorted(reports)]
    base = calibrate.calibrate(raws, factor=1.0, min_baselines=len(raws))   # consistency checks + applicability
    values = {}
    for n in sorted(reports):
        met = reports[n].get("metrics") or {}
        bad = [m for m in METRICS if not _fin(met.get(m))]
        if bad:
            raise LimitsError(f"{n}: {', '.join(bad)} not measured")
        cov = reports[n].get("coverage") or {}
        if not (cov.get("torque") == 1 and cov.get("joint_velocity") == 1):
            raise LimitsError(f"{n}: actuator limits measured on only part of the steps (coverage {cov})")
        values[n] = {m: float(met[m]) for m in METRICS}
    scores = clean_scores(values)
    limit = max(s["score"] for s in scores.values())
    maxima = {m: max(v[m] for v in values.values()) for m in METRICS}
    return {"version": 1, "protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256,
            "rule": "flag when max over metrics of value / max(clean maximum, floor) exceeds the limit",
            "metrics": list(METRICS), "reported_only": list(REPORTED_ONLY), "floors": dict(FLOORS),
            "limit": limit, "clean_maxima": maxima, "clean_values": values, "clean_scores": scores,
            "floors_used_for_new_robots": _floor_used(maxima),
            "sources": base["sources"], "measurement_config": base["measurement_config"],
            "contact_force_samples_per_step": base["contact_force_samples_per_step"],
            "physics_context": base["physics_context"], "command_set": base["command_set"]}


def score_check2(report, limit_doc):
    """A new robot's combined check 2 result: {"status", "flagged", "score", "limit", "driving_metric", ...}."""
    from ..checks import exploits_score
    reasons = exploits_score.applicability(report, limit_doc)
    met, cov = report.get("metrics") or {}, report.get("coverage") or {}
    for m in METRICS:
        if not _fin(met.get(m)):
            reasons.append(f"{m} was not measured")
    if not (cov.get("torque") == 1 and cov.get("joint_velocity") == 1):
        reasons.append(f"actuator limits were measured on only part of the steps (coverage {cov})")
    if reasons:
        return {"check": 2, "status": "invalid", "flagged": None, "reasons": reasons}
    rat = _ratios({m: float(met[m]) for m in METRICS}, limit_doc["clean_maxima"])
    drv = max(rat, key=lambda m: rat[m])
    return {"check": 2, "status": "scored", "flagged": rat[drv] > limit_doc["limit"], "score": rat[drv],
            "limit": limit_doc["limit"], "driving_metric": drv, "ratios": rat,
            "reported_only": {m: met.get(m) for m in REPORTED_ONLY}, "reasons": []}


def leave_one_out(values):
    """Section 6: for each robot, the limit from the other 19 (each scored against 18), and the left-out robot scored
    against the other 19's maxima."""
    out = {}
    for n in values:
        rest = {k: v for k, v in values.items() if k != n}
        lim = max(s["score"] for s in clean_scores(rest).values())
        mx = {m: max(v[m] for v in rest.values()) for m in METRICS}
        rat = _ratios(values[n], mx)
        drv = max(rat, key=lambda m: rat[m])
        out[n] = {"limit_from_others": lim, "score": rat[drv], "driving_metric": drv, "flagged": rat[drv] > lim}
    return out


# ---- sections 4-6: the clean-robot report -------------------------------------------------------------------------

def clean_report(data, success_new, limit_doc, score_check1=None, score_check5=None):
    from ..checks import robustness, transfer
    score_check1 = score_check1 or robustness.score
    score_check5 = score_check5 or transfer.score
    loo = leave_one_out(limit_doc["clean_values"])
    robots, nogo = {}, []
    for n in sorted(data):
        d = data[n]
        rec1 = ((d["state"].get("summary") or {}).get("check1") or {})
        c1_new = score_check1(d["sweep"], success_new)
        c5 = score_check5(d["nominal"], d["newton"], success_new)
        flags = {"check1": bool(rec1.get("flagged")), "check2": bool(loo[n]["flagged"]),
                 "check5": bool(c5.get("flagged"))}
        if any(flags.values()):
            nogo.append(n)
        robots[n] = {
            "check1_recorded": {"flagged": rec1.get("flagged"), "flagged_params": rec1.get("flagged_params"),
                                "cliffs": rec1.get("cliffs")},
            "check1_new_threshold": {"status": c1_new.get("status"), "flagged": c1_new.get("flagged"),
                                     "flagged_params": [k for k, v in (c1_new.get("params") or {}).items()
                                                        if v.get("flagged")]},
            "check2": {"clean_score": limit_doc["clean_scores"][n], "leave_one_out": loo[n],
                       "reported_only": {m: (d["check2"].get("metrics") or {}).get(m) for m in REPORTED_ONLY}},
            "check5": {"status": c5.get("status"), "flagged": c5.get("flagged"), "metrics": c5.get("metrics"),
                       "reasons": c5.get("reasons")},
            "no_go": flags}
    N = len(robots)

    def count(key):
        k = sum(r["no_go"][key] for r in robots.values())
        return {"flagged": k, "of": N, "interval95": wilson(k, N)}
    return {"protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA256,
            "success_threshold_mps": success_new["threshold_mps"], "check2_limit": limit_doc["limit"],
            "counts": {"check1_recorded": count("check1"), "check2_leave_one_out": count("check2"),
                       "check5": count("check5"),
                       "any_no_go": {"flagged": len(nogo), "of": N, "interval95": wilson(len(nogo), N),
                                     "robots": nogo}},
            "check1_new_threshold_flags": sum(bool(r["check1_new_threshold"]["flagged"]) for r in robots.values()),
            "stop_rule": {"stop_if_more_than": STOP_ABOVE, "no_go": len(nogo), "stop": len(nogo) > STOP_ABOVE},
            "robots": robots}


def check_protocol(repo_root):
    p = os.path.join(repo_root, PROTOCOL)
    if os.path.exists(p) and _sha(p) != PROTOCOL_SHA256:
        raise LimitsError(f"{PROTOCOL} differs from the frozen version {PROTOCOL_SHA256[:12]}")


def run(root, out):
    if os.path.exists(out):
        raise LimitsError(f"{out} exists: the limits are written once")
    data = load(root)
    success_new = success_threshold({n: d["nominal"] for n, d in data.items()})
    limit_doc = combined_limit({n: d["check2"] for n, d in data.items()})
    report = clean_report(data, success_new, limit_doc)
    os.makedirs(out)
    for fn, doc in (("frozen_success_p0v5b.json", success_new), ("check2_limit_v2.json", limit_doc),
                    ("layer1_clean_report.json", report)):
        with open(os.path.join(out, fn), "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
    return success_new, limit_doc, report


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.v2.layer1_limits", description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, help="folder holding phase1_clean/ extracted from phase1-clean.tgz")
    p.add_argument("--out", required=True, help="new folder for the three output files")
    p.add_argument("--repo", default=".", help="repository root, to check the frozen protocol's hash")
    a = p.parse_args(argv)
    try:
        check_protocol(a.repo)
        s, lim, rep = run(a.root, a.out)
    except LimitsError as e:
        print(f"[sim2gate] REFUSED: {e}", file=sys.stderr)
        return 2
    c = rep["counts"]
    print(f"[sim2gate] success threshold {s['threshold_mps']:.4f} m/s (reference {s['reference_error_mps']:.4f})")
    print(f"[sim2gate] check 2 combined limit {lim['limit']:.4f}")
    for k in ("check1_recorded", "check2_leave_one_out", "check5", "any_no_go"):
        lo, hi = c[k]["interval95"]
        print(f"[sim2gate] {k}: {c[k]['flagged']} of {c[k]['of']} (95% interval {lo:.2f}-{hi:.2f})")
    print(f"[sim2gate] {'STOP for a decision' if rep['stop_rule']['stop'] else 'stop rule not triggered'} "
          f"({rep['stop_rule']['no_go']} no-go, stop if more than {STOP_ABOVE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
