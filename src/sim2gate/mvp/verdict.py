"""One policy's verdict from checks 2, 1 and 5 (MVP plan, "Report and verdict").

The verdict is no-go if any check flags a risk; otherwise go. Exit codes, as the plan sets them: 0 go, 1 no-go,
2 missing inputs, 3 check invalid (the check's conditions did not hold, e.g. a check 5 parity condition failed or
the policy did not meet nominal success, so that check is not scored and the verdict says why).

When the checks disagree in kind, a flag decides: a policy with a flagged risk is no-go even if another check is
missing or invalid (the risk is real either way, and the verdict lists the other problems). Otherwise missing
inputs (2) come before an invalid check (3). Inputs that measured different checkpoints are refused outright (2).

The verdict carries no policy name: it is keyed by the blinded policy ID only. Every input is recorded by SHA-256.
"""
import hashlib
import json

from ..checks import exploits_score, robustness, transfer

CHECKS = ("2", "1", "5")


def _sha(doc):
    return None if doc is None else hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def _summary(name, res):
    """The parts of a check result the verdict and the acceptance scorer need."""
    s = {"status": res["status"], "flagged": res["flagged"], "reasons": res["reasons"],
         "risks": [r["metric"] for r in res["risks"]]}
    if name == "1":
        s["nominal_success_rate"] = res["nominal_success_rate"]
        s["flagged_params"] = sorted(k for k, v in res["params"].items() if v["flagged"])
        s["cliffs"] = {k: v["cliff"] for k, v in res["params"].items()}
    elif name == "5":
        s["parity"] = res["parity"]
        s.update({k: res["metrics"][k] for k in ("physx_success_rate", "newton_success_rate",
                                                 "newton_fall_free_rate", "error_ratio")})
    return s


def build(policy_id, check2_report=None, sweep_raw=None, physx_raw=None, newton_raw=None, thresholds=None,
          success=None, scorers=None):
    """The verdict document. scorers: {"2": f(report, thresholds), "1": f(raw, success), "5": f(physx, newton,
    success)}; the defaults are the frozen checks (a scratch copy can be passed in for the planted-bug test)."""
    sc = {"2": exploits_score.score, "1": robustness.score, "5": transfer.score}
    sc.update(scorers or {})
    inputs = {"check2_report": check2_report, "sweep_raw": sweep_raw, "physx_raw": physx_raw,
              "newton_raw": newton_raw, "thresholds": thresholds, "success": success}
    need = {"2": ("check2_report", "thresholds"), "1": ("sweep_raw", "success"),
            "5": ("physx_raw", "newton_raw", "success")}
    checks, risks = {}, []
    for name in CHECKS:
        missing = [k for k in need[name] if inputs[k] is None]
        if missing:
            checks[name] = {"status": "missing", "flagged": None, "reasons": [f"missing {', '.join(missing)}"],
                            "risks": []}
            continue
        if name == "2":
            res = sc["2"](check2_report, thresholds)
        elif name == "1":
            res = sc["1"](sweep_raw, success)
        else:
            res = sc["5"](physx_raw, newton_raw, success)
        checks[name] = _summary(name, res)
        risks += res["risks"]
    ck = {"2": (check2_report or {}).get("checkpoint", {}).get("sha256") if check2_report else None,
          "1": (sweep_raw or {}).get("checkpoint_sha256"), "5p": (physx_raw or {}).get("checkpoint_sha256"),
          "5n": (newton_raw or {}).get("checkpoint_sha256")}
    seen = {v for v in ck.values() if v}
    problems = []
    if len(seen) > 1:
        problems.append(f"the checks measured different checkpoints: {ck}")
    if problems:                         # inputs from different policies: nothing here describes one policy
        verdict, code = "incomplete", 2
    elif any(c["flagged"] for c in checks.values()):
        verdict, code = "no-go", 1
    elif any(c["status"] == "missing" for c in checks.values()):
        verdict, code = "incomplete", 2
    elif any(c["status"] == "invalid" for c in checks.values()):
        verdict, code = "invalid", 3
    else:
        verdict, code = "go", 0
    return {"version": 1, "policy_id": policy_id, "verdict": verdict, "exit_code": code,
            "checkpoint_sha256": seen.pop() if len(seen) == 1 else None, "checks": checks, "risks": risks,
            "problems": problems, "inputs_sha256": {k: _sha(v) for k, v in inputs.items()}}


def signature(v):
    """What 'the same verdict' means for the replay criterion (A6): verdict, exit code and the flagged risks."""
    return {"verdict": v["verdict"], "exit_code": v["exit_code"],
            "risks": sorted(f"check {r['check']}: {r['metric']}" for r in v["risks"])}
