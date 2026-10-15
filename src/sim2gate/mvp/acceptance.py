"""Score the blinded run against the answer key: the MVP plan's acceptance criteria A1-A6.

    python -m sim2gate.mvp.acceptance --verdicts <folder of verdict JSONs> --key key.json [--replays <folder>]

Used in step 5, after every verdict is recorded: only then is the key (Vihaan's) opened. The criteria are the
frozen plan's, unchanged:

  A1 check 2 flags P3 and P4 in at least 2 of 3 seeds each
  A2 check 1 flags the friction or mass parameter for P1, and latency for P2, in at least 2 of 3 seeds each
  A3 check 5 flags P5 in at least 2 of 3 seeds
  A4 at most 1 of the 5 held-out baselines (P0 seeds 4 to 8) gets a no-go
  A5 check 5 parity holds for every policy, and every planted policy meets nominal success on PhysX
  A6 rerunning 3 randomly chosen policies reproduces their verdicts exactly

Readings fixed here, before any planted or held-out policy is touched:
- "flags" means the check scored the policy (status "scored") and flagged it; an invalid or missing check flags
  nothing.
- A5's nominal success on PhysX: both of the policy's nominal PhysX measurements, check 1's nominal group and
  check 5's PhysX run, reach 80% episode success. A check 5 that is missing counts as failed parity.
- A6's "verdict": the verdict, its exit code and the list of flagged risks (verdict.signature). The 3 policies
  are drawn with pick_replays from the recorded seed.
- The run is complete only when every policy in the key has a verdict, and no verdict is "incomplete" (exit 2); an
  incomplete run fails acceptance. The key holds the plan's 23 policies: P0 seeds 1-8 and P1-P5 seeds 1-3. P0
  seeds 1-3 (the calibration baselines) are run blinded with the rest and count for A5 and A6; A4 counts only the
  held-out seeds 4-8.

A check that also flags a non-target flaw is reported, not counted as a failure (plan).
"""
import argparse
import glob
import json
import os
import random
import sys

from ..checks.robustness import CLIFF_SUCCESS
from .verdict import signature

EXPECTED = {"P0": (1, 2, 3, 4, 5, 6, 7, 8), "P1": (1, 2, 3), "P2": (1, 2, 3), "P3": (1, 2, 3), "P4": (1, 2, 3),
            "P5": (1, 2, 3)}
PLANTED = ("P1", "P2", "P3", "P4", "P5")
REPLAYS = 3
MIN_SEEDS = 2
HELD_OUT = (4, 5, 6, 7, 8)


def pick_replays(ids, seed, n=REPLAYS):
    """The policies to rerun for A6: a seeded draw over the sorted IDs, so anyone can repeat the choice."""
    return sorted(random.Random(seed).sample(sorted(ids), n))


def _flags(v, check):
    c = v["checks"].get(check) or {}
    return c.get("status") == "scored" and c.get("flagged") is True


def evaluate(verdicts, key, replays=None):
    """verdicts: {policy_id: verdict doc}; key: {policy_id: {"policy": "P3", "seed": 2}}; replays: {policy_id:
    verdict doc of the rerun}. Returns {"passed": bool, "criteria": {...}, "problems": [...], "by_policy": {...}}."""
    problems = []
    seen = {}
    for pid, k in key.items():
        seen.setdefault(k["policy"], []).append(k["seed"])
    for pol, seeds in EXPECTED.items():
        if sorted(seen.get(pol, [])) != list(seeds):
            problems.append(f"the key holds {pol} seeds {sorted(seen.get(pol, []))}, the plan needs {list(seeds)}")
    extra = sorted(set(seen) - set(EXPECTED))
    if extra:
        problems.append(f"the key holds policies outside the plan: {extra}")
    for pid in key:
        if pid not in verdicts:
            problems.append(f"{pid}: no verdict")
        elif verdicts[pid]["exit_code"] == 2:
            problems.append(f"{pid}: incomplete ({'; '.join(verdicts[pid]['problems']) or 'missing inputs'})")
    for pid in verdicts:
        if pid not in key:
            problems.append(f"{pid}: a verdict for a policy that is not in the key")

    def group(pol):
        return sorted(((k["seed"], pid) for pid, k in key.items() if k["policy"] == pol and pid in verdicts))

    def detect(pol, test):
        rows = [{"seed": s, "id": pid, "detected": bool(test(verdicts[pid]))} for s, pid in group(pol)]
        return rows, sum(r["detected"] for r in rows)

    crit = {}
    rows3, n3 = detect("P3", lambda v: _flags(v, "2"))
    rows4, n4 = detect("P4", lambda v: _flags(v, "2"))
    crit["A1"] = {"passed": n3 >= MIN_SEEDS and n4 >= MIN_SEEDS, "P3": rows3, "P4": rows4,
                  "rule": "check 2 flags P3 and P4 in at least 2 of 3 seeds each"}
    rows1, n1 = detect("P1", lambda v: _flags(v, "1") and bool(
        {"ground_friction", "added_base_mass_kg"} & set(v["checks"]["1"].get("flagged_params") or [])))
    rows2, n2 = detect("P2", lambda v: _flags(v, "1") and "action_latency_ms" in (
        v["checks"]["1"].get("flagged_params") or []))
    crit["A2"] = {"passed": n1 >= MIN_SEEDS and n2 >= MIN_SEEDS, "P1": rows1, "P2": rows2,
                  "rule": "check 1 flags friction or mass for P1, and latency for P2, in at least 2 of 3 seeds each"}
    rows5, n5 = detect("P5", lambda v: _flags(v, "5"))
    crit["A3"] = {"passed": n5 >= MIN_SEEDS, "P5": rows5, "rule": "check 5 flags P5 in at least 2 of 3 seeds"}
    rows0 = [{"seed": s, "id": pid, "verdict": verdicts[pid]["verdict"], "risks": signature(verdicts[pid])["risks"]}
             for s, pid in group("P0") if s in HELD_OUT]
    nogo = sum(r["verdict"] == "no-go" for r in rows0)
    crit["A4"] = {"passed": nogo <= 1 and len(rows0) == len(HELD_OUT), "no_go": nogo, "P0": rows0,
                  "rule": "at most 1 of the 5 held-out baselines (P0 seeds 4 to 8) gets a no-go"}
    parity_bad, nominal_bad = [], []
    for pid, k in sorted(key.items()):
        v = verdicts.get(pid)
        if v is None:
            continue
        c5, c1 = v["checks"].get("5") or {}, v["checks"].get("1") or {}
        if c5.get("parity") != "passed":
            parity_bad.append({"id": pid, "reasons": c5.get("reasons")})
        if k["policy"] in PLANTED:
            rates = {"check1_nominal": c1.get("nominal_success_rate"), "check5_physx": c5.get("physx_success_rate")}
            if not all(r is not None and r >= CLIFF_SUCCESS for r in rates.values()):
                nominal_bad.append({"id": pid, **rates})
    crit["A5"] = {"passed": not parity_bad and not nominal_bad, "parity_failed": parity_bad,
                  "nominal_not_met": nominal_bad,
                  "rule": "check 5 parity holds for every policy; every planted policy meets nominal success on PhysX"}
    reps = []
    for pid, rv in sorted((replays or {}).items()):
        if pid not in verdicts:
            reps.append({"id": pid, "same": False, "why": "no original verdict"})
            continue
        a, b = signature(verdicts[pid]), signature(rv)
        reps.append({"id": pid, "same": a == b, "original": a, "replay": b})
    crit["A6"] = {"passed": len(reps) == REPLAYS and all(r["same"] for r in reps), "replays": reps,
                  "rule": "rerunning 3 randomly chosen policies reproduces their verdicts exactly"}
    if len(reps) != REPLAYS:
        crit["A6"]["why"] = f"{len(reps)} replays, the plan needs {REPLAYS}"
    return {"version": 1, "passed": not problems and all(c["passed"] for c in crit.values()), "criteria": crit,
            "problems": problems}


def _load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        v = json.load(open(f))
        if "policy_id" in v:
            out[v["policy_id"]] = v
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.mvp.acceptance", description=__doc__.splitlines()[0])
    p.add_argument("--verdicts", required=True, help="folder of verdict JSON files (one per blinded policy ID)")
    p.add_argument("--key", required=True, help='the answer key: {"policies": {id: {"policy": "P3", "seed": 2}}}')
    p.add_argument("--replays", default=None, help="folder of the A6 rerun verdicts")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    key = json.load(open(a.key))
    res = evaluate(_load_dir(a.verdicts), key.get("policies", key), _load_dir(a.replays) if a.replays else None)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    for name, c in res["criteria"].items():
        print(f"{name} {'PASS' if c['passed'] else 'FAIL'}  {c['rule']}")
    for pr in res["problems"]:
        print(f"  problem: {pr}")
    print("MVP VALIDATION PASSED" if res["passed"] else "MVP VALIDATION FAILED")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
