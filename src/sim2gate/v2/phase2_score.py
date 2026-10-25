"""v2 phase 2, part B: the frozen scoring program (PHASE2_PROTOCOL.md sections 7 and 8).

Stage 1 (the key holder, on his own computer): reads the feature table (random IDs only) and the key, builds the
folds from the key inside this program, and for each fold fits every approach on the four training folds only. The
test fold's labels are never passed to any fitting code. It writes out-of-fold predictions per robot ID, and nothing
about folds.

Stage 2 (after the predictions' SHA-256 is recorded and the key released): checks the hashes and computes the scores,
the criteria C2 and C3, and everything reported alongside them.

Approaches (section 7): A behaviour; B ordinary training diagnostics; C = B + rho; Bvar = B + Var(T_k)/Var(A);
Cplacebo = B + as many N(0, 1) features as C adds (seed 0); Bw = B + weighted contributions; Cw = C + weighted
contributions. Question 1 (nearest centroid) runs for all; question 2 (ridge on window 1) for all but A.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys

LABELS = ("Control", "R-slip", "Phi-slip-mild", "Phi-delay")      # also the tie order
CAUSE = {"Control": None, "R-slip": "reward", "Phi-slip-mild": "physics", "Phi-delay": "physics"}
CONTROL_SEEDS = (101, 102, 103, 104, 105)
N_FOLDS = 5
MISSING_MAX = 0.20                  # section 7, preprocessing step 4 (training folds only)
RIDGE = 1.0
SLIP = "A:check2:foot_slip_p95_mps"
LATENCY = "A:check1_cliff_index:action_latency_ms"
QUALIFY_MIN = 0.85
BOOT_N, BOOT_SEED = 10000, 0
C2_MARGIN, C3_ACC_MARGIN = 0.20, 0.10
APPROACHES = ("A", "B", "C", "Bvar", "Cplacebo", "Bw", "Cw")
Q2_APPROACHES = ("B", "C", "Bvar", "Cplacebo", "Bw", "Cw")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def program_sha256():
    return sha256(os.path.abspath(__file__))


# ---- inputs ----------------------------------------------------------------------------------------------------------

def load_table(path):
    t = json.load(open(path))
    cols = t["columns"]
    return cols, {rid: dict(zip(cols, row)) for rid, row in t["rows"].items()}


def load_key(path):
    """{random_id: {"arm": label, "seed": int}}"""
    k = json.load(open(path))
    k = k.get("robots", k)
    bad = [r for r, v in k.items() if v.get("arm") not in LABELS]
    if bad:
        raise ValueError(f"unknown arm in key for {bad[:3]}")
    return k


def folds_from_key(key):
    """Fold i (1-5) holds each arm's i-th robot by seed order, and control 100 + i (section 7). Built only here."""
    folds = {}
    for arm in LABELS:
        ids = sorted((r for r, v in key.items() if v["arm"] == arm), key=lambda r: key[r]["seed"])
        if arm == "Control":
            ids = [r for r in ids if key[r]["seed"] in CONTROL_SEEDS]
            for r in ids:
                folds[r] = CONTROL_SEEDS.index(key[r]["seed"]) + 1
        else:
            if len(ids) != N_FOLDS:
                raise ValueError(f"{arm}: {len(ids)} robots in the key, expected {N_FOLDS}")
            for i, r in enumerate(ids):
                folds[r] = i + 1
    return folds


def approach_columns(cols, approach, window1=False):
    pick = lambda pre: [c for c in cols if c.startswith(pre) and (not window1 or c.endswith(":w1"))]
    b = pick("B:")
    sets = {"A": pick("A:") if not window1 else [], "B": b, "C": b + pick("C:"), "Bvar": b + pick("Bvar:"),
            "Cplacebo": b, "Bw": b + pick("W:"), "Cw": b + pick("C:") + pick("W:")}
    return sets[approach], (len(pick("C:")) if approach == "Cplacebo" else 0)


def placebo(rows, n, tag):
    """n standard-normal features per robot, seed 0, drawn in ID order (section 7)."""
    rng = random.Random(f"{BOOT_SEED}:{tag}")
    return {rid: [rng.gauss(0.0, 1.0) for _ in range(n)] for rid in sorted(rows)}


# ---- preprocessing and models (training folds only) ------------------------------------------------------------------

def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def prepare(train_X, test_X):
    """Section 7: drop features missing for > 20% of training robots; z-score with training mean and sd (n - 1);
    drop zero-sd features; missing -> training mean (0 after z-scoring). Returns z-scored rows and a report."""
    n = len(train_X)
    p = len(next(iter(train_X.values()))) if n else 0
    keep, dropped_missing, dropped_const = [], 0, 0
    stats = []
    for j in range(p):
        xs = [row[j] for row in train_X.values() if _finite(row[j])]
        if n - len(xs) > MISSING_MAX * n:
            dropped_missing += 1
            continue
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
        if not sd > 0:
            dropped_const += 1
            continue
        keep.append(j)
        stats.append((m, sd))
    z = lambda row: [((row[j] if _finite(row[j]) else m) - m) / sd for j, (m, sd) in zip(keep, stats)]
    filled = sum(1 for row in train_X.values() for j in keep if not _finite(row[j]))
    return ({r: z(v) for r, v in train_X.items()}, {r: z(v) for r, v in test_X.items()},
            {"features": len(keep), "dropped_missing": dropped_missing, "dropped_constant": dropped_const,
             "filled_missing_train": filled})


def nearest_centroid(train_Z, train_labels, test_Z):
    cents = {}
    for r, z in train_Z.items():
        cents.setdefault(train_labels[r], []).append(z)
    cents = {lab: [sum(c) / len(c) for c in zip(*v)] if v and v[0] else [] for lab, v in cents.items()}
    out = {}
    for r, z in test_Z.items():
        d = {lab: sum((a - b) ** 2 for a, b in zip(z, cents[lab])) for lab in LABELS if lab in cents}
        best = min(d.values())
        out[r] = next(lab for lab in LABELS if lab in d and d[lab] == best)    # ties: first label in order
    return out


def ridge(train_Z, y, test_Z, lam=RIDGE):
    """Ridge with an unpenalised intercept: w = (X'X + lam I)^-1 X'(y - mean y) on the z-scored features."""
    import numpy as np
    ids = sorted(train_Z)
    ym = sum(y[r] for r in ids) / len(ids)
    X = np.array([train_Z[r] for r in ids], dtype=np.float64).reshape(len(ids), -1)
    if X.shape[1] == 0:
        return {r: ym for r in test_Z}
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ (np.array([y[r] for r in ids]) - ym))
    return {r: float(ym + np.asarray(z, dtype=np.float64) @ w) for r, z in test_Z.items()}


# ---- stage 1 ---------------------------------------------------------------------------------------------------------

def _matrix(rows, cols, extra=None):
    return {r: [rows[r].get(c) for c in cols] + (extra[r] if extra else []) for r in rows}


def out_of_fold(rows, cols, key, folds, subset=None, labels_used=LABELS):
    """Out-of-fold predictions for every approach. `subset`: robots allowed in training and testing."""
    ids = sorted(r for r in folds if (subset is None or r in subset))
    preds = {r: {} for r in ids}
    report = {}
    for approach in APPROACHES:
        for q, window1 in (("q1", False), ("q2", True)):
            if q == "q2" and approach not in Q2_APPROACHES:
                continue
            c, n_placebo = approach_columns(cols, approach, window1)
            extra = placebo(rows, n_placebo, f"{approach}:{q}") if n_placebo else None
            X = _matrix({r: rows[r] for r in ids}, c, extra)
            for f in range(1, N_FOLDS + 1):
                train = [r for r in ids if folds[r] != f and key[r]["arm"] in labels_used]
                test = [r for r in ids if folds[r] == f]
                Ztr, Zte, rep = prepare({r: X[r] for r in train}, {r: X[r] for r in test})
                report[f"{approach}:{q}:fold{f}"] = rep
                if q == "q1":
                    lab = {r: key[r]["arm"] for r in train}                       # training folds only
                    for r, p in nearest_centroid(Ztr, lab, Zte).items():
                        preds[r][f"{approach}:q1"] = p
                else:
                    for target, name in ((SLIP, "slip"), (LATENCY, "latency")):
                        y = {r: _target(rows[r].get(target), name) for r in train}
                        y = {r: v for r, v in y.items() if v is not None}
                        p = ridge({r: Ztr[r] for r in y}, y, Zte) if len(y) > 1 else {r: None for r in test}
                        for r, v in p.items():
                            preds[r][f"{approach}:q2_{name}"] = v
    return preds, report


def _target(v, name):
    if not _finite(v):
        return None
    if name == "slip":
        return math.log(v) if v > 0 else None
    return float(v)


def stage1(features_path, key_path, out_path):
    cols, rows = load_table(features_path)
    key = load_key(key_path)
    missing = sorted(set(key) - set(rows))
    if missing:
        raise ValueError(f"{len(missing)} robots in the key have no feature row")
    folds = folds_from_key(key)
    primary, rep1 = out_of_fold(rows, cols, key, folds)
    no_delay, rep2 = out_of_fold(rows, cols, key, folds, labels_used=("Control", "R-slip", "Phi-slip-mild"))
    qualified = {r for r in folds if _finite(rows[r].get("A:nominal_success")) and
                 rows[r]["A:nominal_success"] >= QUALIFY_MIN}
    secondary, rep3 = out_of_fold(rows, cols, key, folds, subset=qualified)
    out = {"format": "sim2gate v2 phase 2 predictions 1", "program_sha256": program_sha256(),
           "features_sha256": sha256(features_path),
           "primary": primary, "without_phi_delay": {r: {k: v for k, v in p.items() if k.endswith(":q1")}
                                                      for r, p in no_delay.items()},
           "secondary_qualified_only": secondary,
           "preprocessing": {"primary": rep1, "without_phi_delay": rep2, "secondary": rep3}}
    # Nothing about folds, arms or seeds is written; robots appear in ID order.
    out["primary"] = dict(sorted(out["primary"].items()))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    return out


# ---- stage 2 ---------------------------------------------------------------------------------------------------------

def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def accuracy(preds, key, ids, approach, which="q1"):
    ok = [preds[r].get(f"{approach}:{which}") == key[r]["arm"] for r in ids]
    return sum(ok) / len(ok) if ok else None


def cause_accuracy(preds, key, ids, approach):
    ids = [r for r in ids if CAUSE[key[r]["arm"]]]
    ok = [CAUSE.get(preds[r].get(f"{approach}:q1")) == CAUSE[key[r]["arm"]] for r in ids]
    return sum(ok) / len(ok) if ok else None


def confusion(preds, key, ids, approach):
    m = {t: {p: 0 for p in LABELS} for t in LABELS}
    for r in ids:
        m[key[r]["arm"]][preds[r][f"{approach}:q1"]] += 1
    return m


def q2_spearman(preds, rows, ids, approach, name="slip", target=SLIP):
    pairs = [(preds[r].get(f"{approach}:q2_{name}"), _target(rows[r].get(target), name)) for r in ids]
    pairs = [(p, y) for p, y in pairs if p is not None and y is not None]
    return spearman([p for p, _ in pairs], [y for _, y in pairs]) if pairs else None


def paired_bootstrap(preds, rows, key, ids, a="C", b="B", n=BOOT_N, seed=BOOT_SEED):
    """Spearman(a) - Spearman(b) on fixed out-of-fold predictions; robots resampled within label strata, each robot's
    a and b predictions kept together; undefined resamples redrawn and counted (section 8)."""
    rng = random.Random(seed)
    strata = {}
    for r in ids:
        strata.setdefault(key[r]["arm"], []).append(r)
    diffs, redrawn = [], 0
    while len(diffs) < n:
        draw = [rng.choice(g) for g in strata.values() for _ in g]
        sa, sb = q2_spearman(preds, rows, draw, a), q2_spearman(preds, rows, draw, b)
        if sa is None or sb is None:
            redrawn += 1
            if redrawn > 10 * n:
                raise ValueError("bootstrap: too many undefined resamples")
            continue
        diffs.append(sa - sb)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n)], diffs[int(math.ceil(0.975 * n)) - 1]
    return {"low": lo, "high": hi, "resamples": n, "redrawn_undefined": redrawn}


def stage2(features_path, predictions_path, key_path, predictions_sha256, out_path=None):
    got = sha256(predictions_path)
    if got != predictions_sha256:
        raise ValueError(f"predictions file hash {got} does not match the recorded {predictions_sha256}")
    pred = json.load(open(predictions_path))
    if pred["features_sha256"] != sha256(features_path):
        raise ValueError("the predictions were made from a different feature table")
    if pred["program_sha256"] != program_sha256():
        raise ValueError("stage 1 ran a different scoring program")
    cols, rows = load_table(features_path)
    key = load_key(key_path)
    P = pred["primary"]
    ids = sorted(P)
    acc = {a: accuracy(P, key, ids, a) for a in APPROACHES}
    sp = {a: q2_spearman(P, rows, ids, a) for a in Q2_APPROACHES}
    boot_cb = paired_bootstrap(P, rows, key, ids, "C", "B")
    boot_cv = paired_bootstrap(P, rows, key, ids, "C", "Bvar")

    def beats(x, y, boot):
        by_acc = acc[x] - acc[y] >= C3_ACC_MARGIN - 1e-12
        by_ew = sp[x] is not None and sp[y] is not None and sp[x] > sp[y] and boot["low"] > 0
        return bool(by_acc or by_ew), {"accuracy_gain": acc[x] - acc[y], "early_warning": {
            "spearman": [sp[x], sp[y]], "ci95_difference": [boot["low"], boot["high"]]}}
    c2 = max(acc["B"], acc["C"]) - acc["A"] >= C2_MARGIN - 1e-12
    c3, c3_detail = beats("C", "B", boot_cb)
    c_vs_bvar, cv_detail = beats("C", "Bvar", boot_cv)
    if not c3:
        c3_reading = "C3 not met."
    elif c_vs_bvar:
        c3_reading = "C3 met, and C also beats B-var: the gain is credited to rho."
    else:
        c3_reading = "C3 met, but no demonstrated advantage over the variance baseline."
    slip_ids = [r for r in ids if key[r]["arm"] in ("R-slip", "Phi-slip-mild")]
    nd = pred["without_phi_delay"]
    nd_ids = [r for r in ids if key[r]["arm"] != "Phi-delay"]
    S = pred["secondary_qualified_only"]
    out = {
        "program_sha256": program_sha256(), "predictions_sha256": got, "features_sha256": pred["features_sha256"],
        "robots": len(ids), "chance_accuracy": 1 / len(LABELS),
        "C2": {"pass": bool(c2), "rule": "B's or C's accuracy at least 20 points above A's",
               "accuracy": {a: acc[a] for a in ("A", "B", "C")}},
        "C3": {"pass": c3, "rule": "C at least 10 points above B in accuracy, or C's early-warning Spearman above B's "
                                   "with the 95% paired-bootstrap interval of the difference above zero",
               **c3_detail, "reading": c3_reading,
               "C_against_Bvar": {"beats": c_vs_bvar, **cv_detail}},
        "reported": {
            "accuracy_all_approaches": acc,
            "reward_vs_physics_accuracy": {a: cause_accuracy(P, key, ids, a) for a in APPROACHES},
            "slip_pair_accuracy": {a: accuracy(P, key, slip_ids, a) for a in APPROACHES},
            "confusion": {a: confusion(P, key, ids, a) for a in APPROACHES},
            "early_warning_slip_spearman": sp,
            "early_warning_latency_spearman": {a: q2_spearman(P, rows, ids, a, "latency", LATENCY)
                                               for a in Q2_APPROACHES},
            "question1_without_phi_delay": {a: accuracy(nd, key, nd_ids, a) for a in APPROACHES},
            "secondary_qualified_only": {"robots": len(S), "note": "conditional on qualification",
                                         "accuracy": {a: accuracy(S, key, sorted(S), a) for a in APPROACHES}},
            "bootstrap": {"C_vs_B": boot_cb, "C_vs_Bvar": boot_cv},
        },
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.v2.phase2_score")
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("stage1", help="key holder: out-of-fold predictions")
    s1.add_argument("--features", required=True)
    s1.add_argument("--key", required=True)
    s1.add_argument("--out", required=True)
    s2 = sub.add_parser("stage2", help="after the key is released: scores and criteria")
    s2.add_argument("--features", required=True)
    s2.add_argument("--predictions", required=True)
    s2.add_argument("--predictions-sha256", required=True)
    s2.add_argument("--key", required=True)
    s2.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "stage1":
        stage1(a.features, a.key, a.out)
        print(f"[sim2gate] stage 1 predictions written: {a.out}, sha256 {sha256(a.out)}", flush=True)
        return 0
    r = stage2(a.features, a.predictions, a.key, a.predictions_sha256, a.out)
    print(f"[sim2gate] C2 {'PASS' if r['C2']['pass'] else 'not met'}; C3 {'PASS' if r['C3']['pass'] else 'not met'}"
          f" ({r['C3']['reading']}); accuracy {json.dumps(r['C2']['accuracy'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
