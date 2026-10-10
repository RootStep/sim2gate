"""Set check 2 thresholds from calibration baselines, by the MVP plan's rule: for each metric, the largest value seen
across the clean baselines (P0 seeds 1 to 3), times a safety factor (1.5).

    python -m sim2gate.checks.calibrate --out thresholds_go2_flat.json p0_s1.json p0_s2.json p0_s3.json

Inputs are check 2 reports (sim2gate_check2.json) of the baseline policies. The output is the versioned file that
`python -m sim2gate.checks.isaaclab_exploits --thresholds` reads; it records every input report's SHA-256 and the
policy checkpoint each one measured, so a threshold can be traced to the exact policies that set it.

Refuses (exit 2) rather than guessing when: fewer than --min-baselines reports; a report has no recorded checkpoint;
two reports measured the same checkpoint; the reports were measured with different settings (control step, foot
radius, contact threshold, flat ground, number of steps or envs); or any metric is missing or not finite.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys

from .isaaclab_exploits import CALIBRATED_METRICS

SAME_CONFIG = ("dt", "contact_force_n", "foot_radius_m", "penetration_tol_m", "jitter_window", "flat_ground")


class CalibrationError(ValueError):
    pass


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def calibrate(reports, factor=1.5, min_baselines=3, require_checkpoint=True):
    """reports: list of (path, raw bytes). Returns the thresholds document (dict)."""
    if not _fin(factor) or factor < 1:
        raise CalibrationError(f"factor must be a finite number >= 1, got {factor!r}")
    if len(reports) < min_baselines:
        raise CalibrationError(f"{len(reports)} baseline report(s); the plan needs at least {min_baselines}")
    docs, sources = [], []
    for path, raw in reports:
        try:
            r = json.loads(raw)
        except ValueError as e:
            raise CalibrationError(f"{path}: not JSON ({e})")
        if not isinstance(r, dict) or "metrics" not in r or "config" not in r:
            raise CalibrationError(f"{path}: not a check 2 report")
        ck = r.get("checkpoint")
        if require_checkpoint and not (isinstance(ck, dict) and ck.get("sha256")):
            raise CalibrationError(f"{path}: no recorded checkpoint, so the baseline policy is not identified")
        docs.append(r)
        sources.append({"report": os.path.abspath(path), "report_sha256": hashlib.sha256(raw).hexdigest(),
                        "checkpoint": (ck or {}).get("path"), "checkpoint_sha256": (ck or {}).get("sha256"),
                        "env_steps": r.get("env_steps")})
    shas = [s["checkpoint_sha256"] for s in sources if s["checkpoint_sha256"]]
    if len(set(shas)) != len(shas):
        raise CalibrationError("two reports measured the same checkpoint; each baseline must be a different policy")
    for k in SAME_CONFIG:
        vals = {json.dumps(d["config"].get(k)) for d in docs}
        if len(vals) > 1:
            raise CalibrationError(f"reports were measured with different {k}: {sorted(vals)}")
    for k in ("control_steps", "env_steps"):
        vals = {d.get(k) for d in docs}
        if len(vals) > 1:
            raise CalibrationError(f"reports differ in {k}: {sorted(vals, key=str)}")
    thresholds, maxima = {}, {}
    for m in CALIBRATED_METRICS:
        vals = [d["metrics"].get(m) for d in docs]
        bad = [s["report"] for s, v in zip(sources, vals) if not _fin(v)]
        if bad:
            raise CalibrationError(f"metric {m} missing or not finite in: {', '.join(bad)}")
        maxima[m] = max(vals)
        thresholds[m] = factor * maxima[m]
    try:
        import importlib.metadata as md
        version = md.version("sim2gate")
    except Exception:
        version = None
    return {"version": 1,
            "calibration": f"max over {len(docs)} baselines x{factor}",
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "rule": {"statistic": "max", "factor": factor, "baselines": len(docs)},
            "baseline_maxima": maxima, "thresholds": thresholds, "sources": sources,
            "measurement_config": {k: docs[0]["config"].get(k) for k in SAME_CONFIG},
            "sim2gate_version": version}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.checks.calibrate", description=__doc__.splitlines()[0])
    p.add_argument("reports", nargs="+", help="check 2 reports of the calibration baselines")
    p.add_argument("--out", required=True)
    p.add_argument("--factor", type=float, default=1.5)
    p.add_argument("--min-baselines", type=int, default=3)
    args = p.parse_args(argv)
    try:
        doc = calibrate([(f, open(f, "rb").read()) for f in args.reports], args.factor, args.min_baselines)
    except (CalibrationError, OSError) as e:
        print(f"[sim2gate] calibration refused: {e}", file=sys.stderr)
        return 2
    if os.path.exists(args.out):
        print(f"[sim2gate] {args.out} exists; not overwriting a thresholds file", file=sys.stderr)
        return 2
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    for m, t in doc["thresholds"].items():
        print(f"  {m:40s} max {doc['baseline_maxima'][m]:.6g}  ->  threshold {t:.6g}")
    print(f"[sim2gate] thresholds: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
