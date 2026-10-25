"""v2 phase 2, part B: the feature exporter (PHASE2_PROTOCOL.md sections 5-7) and part A's leak audit.

The exporter turns one robot's records into features, with nothing that identifies the robot:

- A, behaviour: nominal success, check 2's measures, check 1's cliff index per condition, check 5's Newton
  fall-free share and error ratio;
- B, ordinary training diagnostics per window: each reward term's UNWEIGHTED component (the recorder's weighted
  per-step mean divided by weight x step_dt, computed here while the config is at hand), and the trainer's value
  loss, policy entropy (RSL-RL's Gaussian: a monotone function of action std), learning rate, KL and clip fraction;
- C's extra features: rho_k per term per window, from the recorder's per-update moments (section 6's rules);
- B-var's extra features: Var(T_k)/Var(A) per term per window;
- W (secondary experiment only, B_w and C_w): the WEIGHTED per-term means per window.

Excluded from every output: reward weights, configs, run folders, task names, seeds, file names (section 5).

The leak audit (PHASE2_PART_A.md, "Leak audit") runs on the pilot robots' records, whose arms are known. It is not
part of scoring and never sees a study robot.
"""
import argparse
import json
import math
import os
import secrets
import sys

WINDOWS = ((0, 399), (400, 999), (1000, 1999))
LOG_KEYS = (("value_loss", ("losses", "value")), ("entropy", ("losses", "entropy")),
            ("learning_rate", ("learning_rate",)), ("kl", ("kl_subset",)), ("clip_fraction", ("clip_fraction_subset",)))
CHECK2_KEYS = ("foot_slip_p95_mps", "ground_penetration_m", "actuator_limit_step_share",
               "action_jitter_high_freq_power_share", "contact_force_p99_bw", "flight_share_low_speed",
               "base_vz_rms_low_speed_mps")
CHECK1_PARAMS = ("ground_friction", "added_base_mass_kg", "motor_strength_scale", "action_latency_ms",
                 "observation_noise_scale")
NO_CLIFF = 10
CONST_REL = 1e-10          # section 6, rules 2 and 3
RHO_MARGIN = 1e-6          # section 6, rule 4
IDENTITY_REL = 1e-4        # section 6: identity_residual <= 1e-4 x max(1, sd(A)), or the update is left out
MISSING_WINDOW_SHARE = 0.5
FEATURE_PREFIXES = ("A:", "B:", "C:", "Bvar:", "W:")


# ---- reading -----------------------------------------------------------------------------------------------------

def read_updates(evidence_dir):
    meta = json.load(open(os.path.join(evidence_dir, "meta.json")))
    ups = [json.loads(x) for x in open(os.path.join(evidence_dir, "updates.jsonl")) if x.strip()]
    return meta, ups


def reward_weights(env_yaml):
    from .phase2 import load_yaml
    rw = load_yaml(env_yaml).get("rewards") or {}
    return {k: float(v["weight"]) for k, v in rw.items() if isinstance(v, dict) and v.get("weight") is not None}


def _get(d, path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _mean(xs):
    xs = [x for x in xs if _finite(x)]
    return sum(xs) / len(xs) if xs else None


# ---- section 6: rho_k and the variance ratio from one update's moments -------------------------------------------

def rho_and_ratio(update, k):
    """(rho_k, Var(T_k)/Var(A), status) for term index k; status 'ok', 'undefined' or 'invalid'."""
    m, c = _get(update, ("moments", "mean")), _get(update, ("moments", "cov"))
    if not m or not c:
        return None, None, "invalid"
    j = 1 + k
    try:
        va, vt, cat, ma, mt = float(c[0][0]), float(c[j][j]), float(c[0][j]), float(m[0]), float(m[j])
    except (IndexError, TypeError, ValueError):
        return None, None, "invalid"
    if not all(math.isfinite(x) for x in (va, vt, cat, ma, mt)):
        return None, None, "invalid"
    ident = update.get("identity_residual")
    if not _finite(ident) or ident > IDENTITY_REL * max(1.0, math.sqrt(max(va, 0.0))):
        return None, None, "invalid"
    m2a = va + ma * ma
    if m2a <= 0 or va <= CONST_REL * m2a:
        return None, None, "undefined"
    ratio = vt / va
    vd = va - 2 * cat + vt
    m2d = vd + (ma - mt) ** 2
    if vd <= CONST_REL * max(m2d, va):
        return None, ratio, "undefined"
    rho = (va - cat) / math.sqrt(va * vd)
    if abs(rho) > 1 + RHO_MARGIN:
        return None, ratio, "invalid"
    return max(-1.0, min(1.0, rho)), ratio, "ok"


# ---- features --------------------------------------------------------------------------------------------------------

def training_features(meta, updates, weights):
    """B, C-extra, Bvar-extra and W features per window, plus counts for the report (never exported)."""
    names, dt = list(meta["reward_terms"]), meta.get("step_dt")
    feats, counts = {}, {"rho_undefined": 0, "rho_invalid": 0, "rho_checks": 0, "updates": len(updates)}
    for wi, (lo, hi) in enumerate(WINDOWS, 1):
        ups = [u for u in updates if lo <= u.get("iteration", -1) <= hi]
        for k, n in enumerate(names):
            w = weights.get(n)
            wt = [(_get(u, ("term_step_mean",)) or [None] * len(names))[k] for u in ups]
            feats[f"W:wt:{n}:w{wi}"] = _mean(wt)
            if w and _finite(w) and _finite(dt) and dt:
                feats[f"B:unw:{n}:w{wi}"] = _mean([x / (w * dt) if _finite(x) else None for x in wt])
            else:
                feats[f"B:unw:{n}:w{wi}"] = None        # zero-weight term: its unweighted value is not recorded
            rhos, ratios, bad = [], [], 0
            for u in ups:
                r, vr, st = rho_and_ratio(u, k)
                counts["rho_checks"] += 1
                if st != "ok":
                    bad += 1
                    counts["rho_undefined" if st == "undefined" else "rho_invalid"] += 1
                rhos.append(r)
                ratios.append(vr)
            ok = ups and bad <= MISSING_WINDOW_SHARE * len(ups)
            feats[f"C:rho:{n}:w{wi}"] = _mean(rhos) if ok else None
            feats[f"Bvar:vr:{n}:w{wi}"] = _mean(ratios) if ok else None
        for fname, path in LOG_KEYS:
            feats[f"B:log:{fname}:w{wi}"] = _mean([_get(u, path) for u in ups])
    return feats, counts


def cliff_index(name, cliff, envelope="v2"):
    from ..checks import robustness
    if cliff is None:
        return NO_CLIFF
    grid = robustness.values(name, envelope)
    return min(range(len(grid)), key=lambda i: abs(grid[i] - cliff))


def behaviour_features(summary, measure_dir=None):
    """A: from the robot's summary (state.json) and, for check 5, its raw nominal and Newton files."""
    s = summary or {}
    f = {"A:nominal_success": _get(s, ("nominal", "success_rate"))}
    met = s.get("check2_metrics") or {}
    for k in CHECK2_KEYS:
        f[f"A:check2:{k}"] = met.get(k)
    c1 = s.get("check1") or {}
    cl = c1.get("cliffs") or {}
    for p in CHECK1_PARAMS:
        f[f"A:check1_cliff_index:{p}"] = cliff_index(p, cl.get(p)) if c1.get("status") == "scored" else None
    f["A:check5:newton_fall_free"], f["A:check5:error_ratio"] = None, None
    if measure_dir:
        from ..checks import transfer
        pn, nn = os.path.join(measure_dir, "nominal.json"), os.path.join(measure_dir, "newton.json")
        if os.path.exists(pn) and os.path.exists(nn):
            phys, newt = json.load(open(pn)), json.load(open(nn))
            f["A:check5:newton_fall_free"] = transfer.fall_free_rate(newt["episodes"])
            ep, en = transfer._mean_error(phys), transfer._mean_error(newt)
            f["A:check5:error_ratio"] = en / ep if ep and en is not None else None
    return f


def export_robot(evidence_dir, env_yaml, summary=None, measure_dir=None):
    """Every feature of one robot: {name: float or None}, and the report counts. No identifying field."""
    meta, ups = read_updates(evidence_dir)
    t, counts = training_features(meta, ups, reward_weights(env_yaml))
    out = behaviour_features(summary, measure_dir)
    out.update(t)
    return {k: (float(v) if _finite(v) else None) for k, v in out.items()}, counts


def new_id():
    return "r" + secrets.token_hex(6)


def write_table(rows, path):
    """rows: {random_id: features}. Rows sorted by ID; columns sorted; nothing but IDs and numbers."""
    cols = sorted({c for f in rows.values() for c in f})
    bad = [c for c in cols if not c.startswith(FEATURE_PREFIXES)]
    if bad:
        raise ValueError(f"unexpected columns: {bad[:3]}")
    table = {"format": "sim2gate v2 phase 2 features 1", "columns": cols,
             "rows": {rid: [rows[rid].get(c) for c in cols] for rid in sorted(rows)}}
    with open(path, "w") as f:
        json.dump(table, f, indent=0, allow_nan=False)
    return table


# ---- leak audit (part A) --------------------------------------------------------------------------------------------

PRIMARY_PREFIXES = ("B:", "C:", "Bvar:")      # training-side inputs of B, C and B-var


def _same(a, b):
    return a == b or (a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)))


def constant_within_arms(rows, labels):
    """Primary features equal within every arm (2+ robots) yet not equal across all arms: configuration values."""
    groups = {}
    for rid, lab in labels.items():
        groups.setdefault(lab, []).append(rid)
    multi = [g for g in groups.values() if len(g) >= 2]
    found = []
    cols = sorted({c for f in rows.values() for c in f if c.startswith(PRIMARY_PREFIXES)})
    for c in cols:
        vals = {rid: rows[rid].get(c) for rid in rows}
        if all(v is None for v in vals.values()) or not multi:
            continue
        within = all(all(_same(vals[g[0]], vals[r]) for r in g[1:]) for g in multi)
        everywhere = all(_same(vals[next(iter(vals))], v) for v in vals.values())
        if within and not everywhere:
            found.append(c)
    return found


def nearest_centroid_loo(X, labels):
    """Leave-one-robot-out accuracy of nearest centroid on z-scored features (training rows only)."""
    ids = sorted(X)
    hits = 0
    for test in ids:
        train = [i for i in ids if i != test]
        cols = list(range(len(X[test])))
        mu = [_mean([X[i][c] for i in train]) for c in cols]
        sd = []
        for c in cols:
            xs = [X[i][c] for i in train if _finite(X[i][c])]
            m = mu[c]
            v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
            sd.append(math.sqrt(v))
        keep = [c for c in cols if mu[c] is not None and sd[c] > 0]

        def z(i, c):
            x = X[i][c]
            return ((x if _finite(x) else mu[c]) - mu[c]) / sd[c]
        cents = {}
        for i in train:
            cents.setdefault(labels[i], []).append([z(i, c) for c in keep])
        cents = {lab: [sum(col) / len(col) for col in zip(*v)] for lab, v in cents.items()}
        zt = [z(test, c) for c in keep]
        pred = min(sorted(cents), key=lambda lab: sum((a - b) ** 2 for a, b in zip(zt, cents[lab])))
        hits += pred == labels[test]
    return hits / len(ids) if ids else None


def leak_audit(robots):
    """robots: [{"label", "evidence", "env_yaml"}] (pilot robots only). Returns the audit report."""
    rows, labels, terms_seen, weight_ratio, counts = {}, {}, set(), {}, {}
    for i, r in enumerate(robots):
        rid = f"pilot{i:02d}"
        meta, ups = read_updates(r["evidence"])
        terms_seen.add(tuple(meta["reward_terms"]))
        bad_len = sum(len(u.get("term_step_mean") or []) != len(meta["reward_terms"]) for u in ups)
        w = reward_weights(r["env_yaml"])
        feats, cnt = training_features(meta, ups, w)
        cnt["term_length_mismatches"] = bad_len
        rows[rid], labels[rid], counts[rid] = feats, r["label"], cnt
        # the trivial clue: weighted / unweighted per term = weight x step_dt
        weight_ratio[rid] = [(w.get(n) or 0.0) * (meta.get("step_dt") or 0.0) for n in meta["reward_terms"]]
    names_ok = len(terms_seen) == 1 and all(c["term_length_mismatches"] == 0 for c in counts.values())
    const = constant_within_arms(rows, labels)
    acc = nearest_centroid_loo(weight_ratio, labels)
    groups = sorted(set(labels.values()))
    return {
        "robots": len(robots), "arms": {g: sum(v == g for v in labels.values()) for g in groups},
        "1_same_terms_order_format": {"pass": names_ok, "term_lists": [list(t) for t in sorted(terms_seen)]},
        "2_constant_within_arm_features": {"pass": not const, "found": const,
                                           "action": "removed from the primary inputs and reported" if const else None},
        "3_trivial_clue_accuracy": {"accuracy": acc, "chance": 1 / len(groups) if groups else None,
                                    "features": "weight x step_dt per term (recovers the weights)",
                                    "rule": "nearest centroid, leave one robot out", "reported_only": True},
        "rho_counts": {"undefined": sum(c["rho_undefined"] for c in counts.values()),
                       "invalid": sum(c["rho_invalid"] for c in counts.values()),
                       "checks": sum(c["rho_checks"] for c in counts.values())},
        "pass": names_ok,
        "note": "Condition 2 removes features rather than failing the audit (part A). Condition 3 is reported.",
    }


def pilot_robots(works):
    """The pilot robots of the given phase 2 work folders, with their arm labels, from each state.json."""
    from .phase2_arms import all_tasks
    from . import phase1
    out = []
    for work in works:
        st = phase1.load_state(work)
        for name, rec in sorted(st.get("robots", {}).items()):
            if not rec.get("finished") or rec.get("problems"):
                continue
            if rec.get("task") not in all_tasks():
                continue                                   # not a pilot robot (e.g. a control re-measure)
            arm, cid = all_tasks()[rec["task"]]
            ev = os.path.join(work, "robots", name, "evidence")
            y = os.path.join(rec["run_dir"], "params", "env.yaml") if rec.get("run_dir") else None
            if not (y and os.path.exists(y)):
                y = os.path.join(work, "robots", name, "checkpoint", "params", "env.yaml")
            out.append({"label": f"{arm}:{cid}", "evidence": ev, "env_yaml": y, "name": name})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.v2.phase2_export")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="part A's leak audit on the pilot robots")
    a.add_argument("--work", action="append", required=True, help="a pilot work folder (repeat for each round)")
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    robots = pilot_robots(args.work)
    rep = leak_audit(robots)
    rep["robot_names"] = [r["name"] for r in robots]
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    from .phase1 import sha256
    print(f"[sim2gate] leak audit: {rep['robots']} pilot robots; same terms {rep['1_same_terms_order_format']['pass']}; "
          f"constant-within-arm features {len(rep['2_constant_within_arm_features']['found'])}; trivial-clue accuracy "
          f"{rep['3_trivial_clue_accuracy']['accuracy']} (chance {rep['3_trivial_clue_accuracy']['chance']:.3f}); "
          f"rho undefined {rep['rho_counts']['undefined']}, invalid {rep['rho_counts']['invalid']} of "
          f"{rep['rho_counts']['checks']} update-term pairs; sha256 {sha256(args.out)}", flush=True)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
