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


def _details(name, res):
    """What the HTML report shows as evidence for each check."""
    if name == "2":
        return {"metrics": res.get("metrics")}
    if name == "1":
        return {"params": {k: {f: v.get(f) for f in ("series", "cliff", "plausible", "flagged", "unit",
                                                      "training_range")}
                           for k, v in (res.get("params") or {}).items()},
                "success_threshold_mps": res.get("success_threshold_mps")}
    return {"metrics": res.get("metrics"), "rule": res.get("rule"), "parity": res.get("parity")}


def _provenance(check2_report, sweep_raw, physx_raw, newton_raw):
    def pick(raw, *keys):
        return {k: (raw or {}).get(k) for k in keys} if raw else None
    try:
        import importlib.metadata as md
        version = md.version("sim2gate")
    except Exception:
        version = None
    return {"sim2gate_version": version,
            "check2": pick(check2_report, "physics_context", "control_steps", "env_steps") if check2_report else None,
            "sweep": pick(sweep_raw, "task", "env_cfg_sha256", "command_set_sha256", "seed", "overrides"),
            "physx": pick(physx_raw, "env_cfg_sha256", "seed"),
            "newton": pick(newton_raw, "env_cfg_sha256", "seed"),
            "simulator_versions": ((sweep_raw or physx_raw or {}).get("physics") or {}).get("versions")}


def joint_summary(rep):
    """The report-only joint sweep, as the verdict records it (never part of go / no-go)."""
    if rep is None:
        return None
    if rep.get("status") != "scored":
        return {"status": rep.get("status"), "reasons": rep.get("reasons")}
    return {"status": "scored", "report_only": True, "episodes": rep["episodes"], "success_rate": rep["success_rate"],
            "all_realistic": rep["all_realistic"], "any_beyond": rep["any_beyond"],
            "worst_pairs": [p for p in rep["pairs"] if p.get("interaction") is not None][:3]}


def build(policy_id, check2_report=None, sweep_raw=None, physx_raw=None, newton_raw=None, thresholds=None,
          success=None, scorers=None, joint_raw=None):
    """The verdict document. scorers: {"2": f(report, thresholds), "1": f(raw, success), "5": f(physx, newton,
    success)}; the defaults are the frozen checks (a scratch copy can be passed in for the planted-bug test)."""
    sc = {"2": exploits_score.score, "1": robustness.score, "5": transfer.score}
    sc.update(scorers or {})
    inputs = {"check2_report": check2_report, "sweep_raw": sweep_raw, "physx_raw": physx_raw,
              "newton_raw": newton_raw, "thresholds": thresholds, "success": success}
    need = {"2": ("check2_report", "thresholds"), "1": ("sweep_raw", "success"),
            "5": ("physx_raw", "newton_raw", "success")}
    checks, risks, details = {}, [], {}
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
        details[name] = _details(name, res)
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
    joint = None
    if joint_raw is not None and success is not None:
        joint = joint_summary(robustness.joint_report(joint_raw, success))
        jck = joint_raw.get("checkpoint_sha256")
        if jck and seen and jck not in seen:
            joint = {"status": "invalid", "reasons": ["the joint sweep measured a different checkpoint"]}
    return {"version": 1, "policy_id": policy_id, "verdict": verdict, "exit_code": code,
            "checkpoint_sha256": seen.pop() if len(seen) == 1 else None, "checks": checks, "risks": risks,
            "problems": problems, "inputs_sha256": {k: _sha(v) for k, v in inputs.items()},
            "report_only": {"joint_sweep": joint}, "details": details,
            "provenance": _provenance(check2_report, sweep_raw, physx_raw, newton_raw)}


def signature(v):
    """What 'the same verdict' means for the replay criterion (A6): verdict, exit code and the flagged risks."""
    return {"verdict": v["verdict"], "exit_code": v["exit_code"],
            "risks": sorted(f"check {r['check']}: {r['metric']}" for r in v["risks"])}
