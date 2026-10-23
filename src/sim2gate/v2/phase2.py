"""v2 phase 2 driver, part A: smoke test and arm pilots (docs/v2/PHASE2_PROTOCOL.md, part A).

    smoke   each arm's first candidate trained for 2 rounds, its saved config checked for the planted change; one
            control re-measured on check 2 to confirm the hop measures are written (~20 min)
    pilot   re-measure controls 101-105 on check 2 (hop reference), then each arm's candidates in order, 2 seeds each,
            until one passes: both seeds show the intended symptom and qualify (>= 85% over 256, v2 threshold)
    bundle  one .tgz of the pilot record with a SHA-256 manifest

Inside the Isaac Lab container, from the Isaac Lab-Arena folder:
    python -m sim2gate.v2.phase2 smoke --work v2_runs/phase2_smoke --phase1 v2_runs/phase1_clean
    python -m sim2gate.v2.phase2 pilot --work v2_runs/phase2_pilot --phase1 v2_runs/phase1_clean
"""
import argparse
import glob
import json
import os
import sys

from . import phase1
from .phase2_arms import ARM_ORDER, ARMS, BASE_TASK, candidate, symptom_present, task_id

CONTROL_SEEDS = (101, 102, 103, 104, 105)
PILOT_FIRST_SEED = 201
PILOT_LAST_SEED = 240
PILOT_MEASURE = ("nominal", "check2")
HOP_KEYS = ("flight_share_low_speed", "base_vz_rms_low_speed_mps")


def seed_table():
    """Fixed pilot seeds per (arm, candidate), in the protocol's order, whatever earlier pilots show."""
    out, sd = {}, PILOT_FIRST_SEED
    for arm in ARM_ORDER:
        for c in ARMS[arm]["candidates"]:
            out[(arm, c["id"])] = (sd, sd + 1)
            sd += 2
    if sd - 1 > PILOT_LAST_SEED:
        raise ValueError("pilot seeds exceed 201-240")
    return out


def load_yaml(path):
    import yaml

    class _L(yaml.SafeLoader):
        pass
    _L.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
    _L.add_multi_constructor("", lambda loader, suffix, node: None)
    return yaml.load(open(path).read(), Loader=_L) or {}


def _say(msg):
    print(f"[sim2gate] {msg}", flush=True)


def _success():
    from .layer1_limits import load_success
    return load_success()


# ---- controls: check 2 re-measured with the hop measures ------------------------------------------------------------

def remeasure_controls(work, phase1_work, runner, seeds=CONTROL_SEEDS):
    st1 = phase1.load_state(phase1_work)
    out = {}
    for seed in seeds:
        name = phase1.robot_name(seed)
        r = (st1.get("robots") or {}).get(name) or {}
        ck = (r.get("checkpoint") or {}).get("path")
        if not ck or not os.path.exists(ck):
            raise SystemExit(f"control {name}: checkpoint not found ({ck})")
        d = os.path.join(work, "controls", name)
        os.makedirs(d, exist_ok=True)
        cmds, files = phase1.measure_commands(ck, d, task=BASE_TASK)
        if not os.path.exists(files["check2"]):
            runner(cmds["check2"], os.path.join(d, "check2.log"))
        met = json.load(open(files["check2"])).get("metrics") if os.path.exists(files["check2"]) else None
        out[name] = {"checkpoint_sha256": (r.get("checkpoint") or {}).get("sha256"), "check2_metrics": met}
    return out


def hop_reference(controls):
    ref = {}
    for k in HOP_KEYS:
        vals = [(c.get("check2_metrics") or {}).get(k) for c in controls.values()]
        if any(not isinstance(v, (int, float)) for v in vals):
            raise SystemExit(f"hop reference: {k} missing for a control")
        ref[k] = max(vals)
    return ref


# ---- the saved config shows the planted change ----------------------------------------------------------------------

def check_config(arm, cid, env_yaml):
    """Problems (empty if none) comparing a run's saved params/env.yaml with the arm's candidate."""
    d = load_yaml(env_yaml)
    c, bad = candidate(arm, cid), []
    base = {"feet_slide": -0.25, "action_rate_l2": -0.01, "feet_air_time": 0.25}
    for term, k in (c.get("reward_scale") or {}).items():
        w = ((d.get("rewards") or {}).get(term) or {}).get("weight")
        if w is None or abs(w - base[term] * k) > 1e-9:
            bad.append(f"rewards.{term}.weight is {w}, expected {base[term] * k}")
    if "static_friction" in c:
        pm = ((d.get("events") or {}).get("physics_material") or {}).get("params") or {}
        if list(pm.get("static_friction_range") or []) != list(c["static_friction"]):
            bad.append(f"static friction {pm.get('static_friction_range')}, expected {list(c['static_friction'])}")
    if "max_delay_steps" in c:
        legs = ((((d.get("scene") or {}).get("robot") or {}).get("actuators") or {}).get("base_legs") or {})
        if legs.get("max_delay") != c["max_delay_steps"]:
            bad.append(f"base_legs.max_delay is {legs.get('max_delay')}, expected {c['max_delay_steps']}")
    terms = sorted(k for k, v in (d.get("rewards") or {}).items() if v)
    return bad, terms


def _env_yaml(rec):
    rd = rec.get("run_dir")
    p = os.path.join(rd, "params", "env.yaml") if rd else None
    return p if p and os.path.exists(p) else None


# ---- pilots -----------------------------------------------------------------------------------------------------------

def robot_verdict(rec, symptom, hop_ref):
    s = rec.get("summary") or {}
    nom, met = s.get("nominal") or {}, s.get("check2_metrics") or {}
    q = bool(nom.get("qualifies"))
    sym = symptom_present(symptom, met, hop_ref) if met else False
    return {"qualifies": q, "symptom": sym, "pass": q and sym and not rec.get("problems"),
            "success_rate": nom.get("success_rate")}


def pilot(work, phase1_work, runner):
    os.makedirs(work, exist_ok=True)
    state = phase1.load_state(work)
    if "controls" not in state:
        _say("re-measuring controls 101-105 on check 2 (hop reference)")
        state["controls"] = remeasure_controls(work, phase1_work, runner)
        phase1.save_state(work, state)
    ref = hop_reference(state["controls"])
    state["hop_reference"] = ref
    from .phase2_arms import symptom_thresholds
    state["symptom_thresholds"] = symptom_thresholds(ref)
    _say(f"hop reference (max of controls): {json.dumps(ref)}")
    state.setdefault("arms", {})
    table = seed_table()
    success = _success()
    for arm in ARM_ORDER:
        a = state["arms"].setdefault(arm, {"tried": [], "setting": None})
        for c in ARMS[arm]["candidates"]:
            if a["setting"]:
                break
            seeds = table[(arm, c["id"])]
            entry = next((t for t in a["tried"] if t["candidate"] == c["id"]), None)
            if entry is None:
                entry = {"candidate": c["id"], "seeds": list(seeds), "robots": {}}
                a["tried"].append(entry)
            for sd in seeds:
                name = f"p2pilot_{arm}_{c['id']}_s{sd}".replace(".", "_")
                rec = state["robots"].get(name)
                if not (rec and rec.get("finished") and not rec.get("problems")):
                    _say(f"pilot {name} ({task_id(arm, c['id'])})")
                    rec = phase1.train_and_measure(work, name, sd, runner, evidence=True, measure=PILOT_MEASURE,
                                                   require_qualify=False, task=task_id(arm, c["id"]),
                                                   success=success)
                    y = _env_yaml(rec)
                    bad, terms = check_config(arm, c["id"], y) if y else (["no saved env.yaml"], [])
                    rec["config_check"] = {"problems": bad, "reward_terms": terms}
                    if bad:
                        rec["problems"] += [f"config: {b}" for b in bad]
                    state["robots"][name] = rec
                    phase1.save_state(work, state)
                v = robot_verdict(rec, ARMS[arm]["symptom"], ref)
                entry["robots"][name] = v
                _say(f"  {name}: success {v['success_rate']}, qualifies {v['qualifies']}, symptom {v['symptom']}")
            entry["pass"] = all(v["pass"] for v in entry["robots"].values()) and len(entry["robots"]) == 2
            if entry["pass"]:
                a["setting"] = c["id"]
            phase1.save_state(work, state)
        a["dropped"] = a["setting"] is None
        _say(f"{arm}: {'setting ' + a['setting'] if a['setting'] else 'DROPPED (no candidate passed)'}")
    terms = {tuple(r.get("config_check", {}).get("reward_terms") or []) for r in state["robots"].values()}
    state["same_reward_terms"] = len(terms) == 1
    state["result"] = {arm: state["arms"][arm]["setting"] for arm in ARM_ORDER}
    phase1.save_state(work, state)
    _say(f"PILOTS COMPLETE: {json.dumps(state['result'])}; same reward terms in every pilot: {state['same_reward_terms']}")
    return 0


# ---- smoke --------------------------------------------------------------------------------------------------------

def smoke(work, phase1_work, runner):
    os.makedirs(work, exist_ok=True)
    problems = []
    for arm in ARM_ORDER:
        cid = ARMS[arm]["candidates"][0]["id"]
        name = f"smoke_{arm}_{cid}".replace(".", "_")
        rec = phase1.train_and_measure(work, name, 99, runner, evidence=False, measure=(), iterations=2,
                                       task=task_id(arm, cid), success=_success())
        y = _env_yaml(rec)
        bad, terms = check_config(arm, cid, y) if y else (["no saved env.yaml"], [])
        _say(f"smoke {arm}:{cid}: config {'OK' if not bad else bad}; reward terms {len(terms)}")
        problems += [f"{arm}: {b}" for b in bad] + [f"{arm}: {p}" for p in rec.get("problems") or []
                                                     if "no output" not in p]
    ctl = remeasure_controls(os.path.join(work, "control"), phase1_work, runner, seeds=(101,))
    met = next(iter(ctl.values()))["check2_metrics"] or {}
    for k in HOP_KEYS:
        if not isinstance(met.get(k), (int, float)):
            problems.append(f"control check 2: {k} not written")
    _say(f"control check 2 hop measures: {json.dumps({k: met.get(k) for k in HOP_KEYS})}")
    _say("SMOKE OK" if not problems else "SMOKE FAILED: " + " | ".join(problems))
    return 0 if not problems else 1


def main(argv=None):
    from ..mvp.testset import quiet_run
    p = argparse.ArgumentParser(prog="python -m sim2gate.v2.phase2", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("smoke", "pilot", "bundle"):
        s = sub.add_parser(c)
        s.add_argument("--work", required=True)
        if c in ("smoke", "pilot"):
            s.add_argument("--phase1", required=True, help="phase 1's work folder (for the control checkpoints)")
        if c == "bundle":
            s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "smoke":
        return smoke(a.work, a.phase1, quiet_run)
    if a.cmd == "pilot":
        return pilot(a.work, a.phase1, quiet_run)
    return phase1.bundle(a.work, a.out)


if __name__ == "__main__":
    sys.exit(main())
