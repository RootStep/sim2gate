"""The planted-bug test of the MVP validation (plan, "Can the validation fail?").

    python -m sim2gate.mvp.planted_bugs [--out result.json]

Before the blinded run, plant one bug in a scratch copy of each check and confirm the validation catches it. For
each check this plants two: one that hides real flags (a missed detection) and one that raises false alarms. Each
bug is a one-line edit to a copy of the check's source file, imported under its own name; the checks themselves are
never modified. The copy then scores a stand-in answer-key set through the same pipeline the blinded run uses
(sim2gate.mvp.verdict, then sim2gate.mvp.acceptance A1-A6). A bug is caught when the validation fails on the
criterion that bug should break. As a control, unedited scratch copies of all three checks must pass every
criterion on the same set.

The stand-in set has the plan's shape (P0 seeds 1-8, P1-P5 seeds 1-3; 23 blinded IDs) and is built from measured
data in this repository. It is a test fixture, not a result about any planted policy:
- P0 1-8: the P0 v4 calibration baselines' own measurements (seeds 1, 2, 3, then 1, 2, 3, 1, 2 again), real.
- P2: the P0 v2 policies, trained without any command delay (P2's planted flaw), real: check 1 flags their latency
  on all three seeds (the real check, unmodified).
- P1, P3, P4, P5: no such policies exist yet, so each is a P0 v4 measurement with its target signal edited in
  (synthetic): P1 fails every episode at friction 0.6 and below; P3's foot slip and P4's action jitter are doubled;
  P5 falls in 2 of every 5 Newton episodes.
"""
import argparse
import copy
import importlib.util
import json
import os
import random
import sys
import tempfile

from ..checks import exploits_score, robustness, transfer
from . import acceptance, verdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
V4 = os.path.join(REPO, "docs", "mvp-plan", "calibration", "p0v4")
V2_CHECK2 = os.path.join(REPO, "docs", "mvp-plan", "calibration", "p0v2")
V2_RAW = os.path.join(REPO, "docs", "mvp-plan", "evidence", "planted-bugs", "p0v2")
REPLAY_SEED = 20261015
ID_SEED = 20261015

# (id, check, module, line in the check's source, its replacement, what it does, criteria it must break)
BUGS = [
    ("check2-missed", "2", exploits_score, "and v > t\n", "and v > t * 10\n",
     "check 2 thresholds 10x too high", ["A1"]),
    ("check2-false-alarm", "2", exploits_score, "and v > t\n", "and v > t / 10\n",
     "check 2 thresholds 10x too low", ["A4"]),
    ("check1-missed", "1", robustness, "CLIFF_SUCCESS = 0.8 ", "CLIFF_SUCCESS = 0.08 ",
     "check 1 cliff at 8% success instead of 80%", ["A2"]),
    ("check1-false-alarm", "1", robustness, 'flagged = cliff is not None and _inside(cliff["value"], p["plausible"])',
     "flagged = cliff is not None", "check 1 flags any cliff in the sweep, not only inside the realistic range",
     ["A4"]),
    ("check5-missed", "5", transfer, "NEWTON_FALL_FREE_MIN = 0.8\n", "NEWTON_FALL_FREE_MIN = 0.08\n",
     "check 5 Newton fall-free minimum 10x too low", ["A3"]),
    ("check5-false-alarm", "5", transfer, "ERROR_RATIO_MAX = 2.0\n", "ERROR_RATIO_MAX = 0.2\n",
     "check 5 error-ratio limit 10x too low", ["A4"]),
]
ENTRY = {"2": "score", "1": "score", "5": "score"}


def scratch_copy(module, old=None, new=None, tag="clean"):
    """Import a copy of `module`'s source, with `old` replaced by `new` (exactly one occurrence), under its own
    name in the same package, so its relative imports resolve to the real package and the real module is
    untouched."""
    src = open(module.__file__).read()
    if old is not None:
        if src.count(old) != 1:
            raise ValueError(f"{module.__name__}: the planted line occurs {src.count(old)} times, not once: {old!r}")
        src = src.replace(old, new)
    pkg, base = module.__name__.rsplit(".", 1)
    name = f"{pkg}._scratch_{base}_{tag.replace('-', '_')}"
    d = tempfile.mkdtemp(prefix="sim2gate_scratch_")
    path = os.path.join(d, f"{base}.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _j(*p):
    return json.load(open(os.path.join(*p)))


def _v4(s):
    return {"check2_report": _j(V4, f"check2_P0_s{s}.json"), "sweep_raw": _j(V4, "checks15", f"raw_sweep_s{s}.json"),
            "physx_raw": _j(V4, "checks15", f"raw_physx_s{s}.json"),
            "newton_raw": _j(V4, "checks15", f"raw_newton_s{s}.json")}


def _v2(s):
    return {"check2_report": _j(V2_CHECK2, f"check2_P0_s{s}.json"), "sweep_raw": _j(V2_RAW, f"raw_sweep_s{s}.json"),
            "physx_raw": _j(V2_RAW, f"raw_physx_s{s}.json"), "newton_raw": _j(V2_RAW, f"raw_newton_s{s}.json")}


def _p1(d):
    d = copy.deepcopy(d)
    raw = d["sweep_raw"]
    bad = {g["group"] for g in raw["groups"] if g["param"] == "ground_friction" and g["value"] <= 0.6 + 1e-9}
    for e in raw["episodes"]:
        if e["group"] in bad:
            e["fell"] = True
    return d


def _scale(metric, factor):
    def f(d):
        d = copy.deepcopy(d)
        d["check2_report"]["metrics"][metric] *= factor
        return d
    return f


def _p5(d):
    d = copy.deepcopy(d)
    for e in d["newton_raw"]["episodes"]:
        if e["env"] % 5 < 2:
            e["fell"] = True
    return d


def standin_set():
    """(inputs by blinded ID, the answer key, how each stand-in was made)."""
    rows = [("P0", s, _v4(base), "real: P0 v4 seed %d" % base)
            for s, base in zip((1, 2, 3, 4, 5, 6, 7, 8), (1, 2, 3, 1, 2, 3, 1, 2))]
    rows += [("P2", s, _v2(s), "real: P0 v2 seed %d (trained without delay)" % s) for s in (1, 2, 3)]
    for pol, edit, what in (("P1", _p1, "every episode fails at friction 0.6 and below"),
                            ("P3", _scale("foot_slip_p95_mps", 2.0), "foot slip doubled"),
                            ("P4", _scale("action_jitter_high_freq_power_share", 2.0), "action jitter doubled"),
                            ("P5", _p5, "2 in 5 Newton episodes fall")):
        rows += [(pol, s, edit(_v4(s)), f"synthetic: P0 v4 seed {s}, {what}") for s in (1, 2, 3)]
    ids = [f"X{n:02d}" for n in random.Random(ID_SEED).sample(range(10, 100), len(rows))]
    inputs, key, made = {}, {}, {}
    for pid, (pol, seed, d, how) in zip(ids, rows):
        inputs[pid], key[pid], made[pid] = d, {"policy": pol, "seed": seed}, how
    return inputs, key, made


def validate(inputs, key, scorers=None):
    th, su = _j(V4, "thresholds.json"), _j(V4, "checks15", "success.json")
    verdicts = {pid: verdict.build(pid, thresholds=th, success=su, scorers=scorers, **d) for pid, d in inputs.items()}
    replays = {pid: verdict.build(pid, thresholds=th, success=su, scorers=scorers, **inputs[pid])
               for pid in acceptance.pick_replays(inputs, REPLAY_SEED)}
    return acceptance.evaluate(verdicts, key, replays), verdicts


def run():
    inputs, key, made = standin_set()
    clean = {c: getattr(scratch_copy(m, tag=f"clean{c}"), ENTRY[c])
             for c, m in (("2", exploits_score), ("1", robustness), ("5", transfer))}
    res, _ = validate(inputs, key, clean)
    out = {"version": 1, "standins": {pid: {**key[pid], "made_from": made[pid]} for pid in sorted(key)},
           "control": {"passed": res["passed"], "criteria": {k: v["passed"] for k, v in res["criteria"].items()},
                       "problems": res["problems"]},
           "bugs": []}
    for bid, check, module, old, new, what, expect in BUGS:
        mod = scratch_copy(module, old, new, tag=bid)
        scorers = dict(clean)
        scorers[check] = getattr(mod, ENTRY[check])
        r, _ = validate(inputs, key, scorers)
        failed = sorted(k for k, v in r["criteria"].items() if not v["passed"])
        out["bugs"].append({"id": bid, "check": check, "bug": what, "edit": {"from": old.strip(), "to": new.strip()},
                            "validation_passed": r["passed"], "failed_criteria": failed, "must_fail": expect,
                            "caught": (not r["passed"]) and set(expect) <= set(failed)})
    out["passed"] = out["control"]["passed"] and all(b["caught"] for b in out["bugs"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.mvp.planted_bugs", description=__doc__.splitlines()[0])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = run()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    c = out["control"]
    print(f"control (unedited copies): validation {'PASSED' if c['passed'] else 'FAILED'} "
          f"{' '.join(k + ('' if v else '!') for k, v in c['criteria'].items())}")
    for pr in c["problems"]:
        print(f"  problem: {pr}")
    for b in out["bugs"]:
        print(f"{b['id']:20s} {b['bug']:75s} failed {','.join(b['failed_criteria']) or '-':12s} "
              f"{'CAUGHT' if b['caught'] else 'NOT CAUGHT'}")
    print("PLANTED-BUG TEST PASSED" if out["passed"] else "PLANTED-BUG TEST FAILED")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
