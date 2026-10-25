"""v2 phase 2 exporter and leak audit, on synthetic training records. No Isaac Lab."""
import json
import math
import random

import pytest

from sim2gate.v2 import phase2_export as ex

TERMS = ["track_lin_vel_xy_exp", "feet_slide", "action_rate_l2", "dof_pos_limits"]
BASE_W = {"track_lin_vel_xy_exp": 1.5, "feet_slide": -0.25, "action_rate_l2": -0.01, "dof_pos_limits": 0.0}
DT = 0.02


def moments(rng, k=len(TERMS)):
    """Mean and covariance of [A, T_1..T_K, V] built from real vectors so the identity A = sum T + V holds."""
    n = 200
    T = [[rng.gauss(0, 1 + j) for _ in range(n)] for j in range(k)]
    V = [rng.gauss(0, 0.5) for _ in range(n)]
    A = [sum(T[j][i] for j in range(k)) + V[i] for i in range(n)]
    cols = [A] + T + [V]
    mu = [sum(c) / n for c in cols]
    cov = [[sum((a - mu[x]) * (b - mu[y]) for a, b in zip(cols[x], cols[y])) / (n - 1) for y in range(len(cols))]
           for x in range(len(cols))]
    return {"mean": mu, "cov": cov}


def make_robot(tmp, name, weights, behaviour_seed=0, iterations=(0, 100, 450, 900, 1200, 1999)):
    """A robot whose BEHAVIOUR (unweighted per-step term values, moments of the unweighted parts) is fixed by
    behaviour_seed; its weighted records follow from `weights`."""
    rng = random.Random(behaviour_seed)
    ev = tmp / name / "evidence"
    ev.mkdir(parents=True)
    json.dump({"reward_terms": TERMS, "step_dt": DT, "seed": 301, "task": "Sim2Gate-...-R-slip-k0.1"},
              open(ev / "meta.json", "w"))
    with open(ev / "updates.jsonl", "w") as f:
        for it in iterations:
            unweighted = [rng.uniform(0.1, 1.0) for _ in TERMS]
            f.write(json.dumps({
                "iteration": it,
                "term_step_mean": [u * weights[t] * DT for u, t in zip(unweighted, TERMS)],
                "moments": moments(rng), "identity_residual": 1e-7,
                "losses": {"value": rng.random(), "entropy": rng.random(), "surrogate": 0.0},
                "learning_rate": 1e-3, "kl_subset": 0.01, "clip_fraction_subset": 0.1}) + "\n")
    y = tmp / name / "env.yaml"
    y.write_text("rewards:\n" + "".join(f"  {t}: {{weight: {weights[t]}}}\n" for t in TERMS)
                 + "seed: 301\nexperiment_name: secret_run_name\n")
    return str(ev), str(y)


def test_unweighted_components_ignore_a_rescaled_weight(tmp_path):
    ev1, y1 = make_robot(tmp_path, "a", BASE_W, behaviour_seed=7)
    w = dict(BASE_W, feet_slide=BASE_W["feet_slide"] * 0.1)              # R-slip's change, behaviour held fixed
    ev2, y2 = make_robot(tmp_path, "b", w, behaviour_seed=7)
    f1, _ = ex.export_robot(ev1, y1)
    f2, _ = ex.export_robot(ev2, y2)
    for c in f1:
        if c.startswith("B:unw:"):
            assert f1[c] == pytest.approx(f2[c], rel=1e-12) or (f1[c] is None and f2[c] is None)
    assert f1["W:wt:feet_slide:w1"] == pytest.approx(10 * f2["W:wt:feet_slide:w1"])   # the weighted ones do move
    assert f1["B:unw:dof_pos_limits:w1"] is None                                       # zero weight: not recorded


def test_no_excluded_field_reaches_the_table(tmp_path):
    ev, y = make_robot(tmp_path, "p2pilot_R-slip_k0_1_s301", BASE_W)
    f, _ = ex.export_robot(ev, y, summary={"nominal": {"success_rate": 0.97}})
    rows = {ex.new_id(): f}
    out = tmp_path / "features.json"
    ex.write_table(rows, out)
    t = json.loads(out.read_text())
    assert set(t) == {"format", "columns", "rows"} and all(c.startswith(ex.FEATURE_PREFIXES) for c in t["columns"])
    assert list(t["rows"]) == list(rows)                                      # the only key: the random ID
    assert all(v is None or isinstance(v, float) for row in t["rows"].values() for v in row)   # numbers only
    names = " ".join(t["columns"]) + " " + t["format"]
    for bad in ("301", "R-slip", "secret_run_name", "p2pilot", "weight", "seed", "env.yaml", "task", str(tmp_path)):
        assert bad not in names
    with pytest.raises(ValueError):
        ex.write_table({"rx": {"seed": 1.0}}, tmp_path / "bad.json")


def test_rows_are_sorted_by_random_id_not_input_order(tmp_path):
    ids = [ex.new_id() for _ in range(6)]
    assert len(set(ids)) == 6 and all(len(i) == 13 for i in ids)
    t = ex.write_table({i: {"A:x": float(n)} for n, i in enumerate(ids)}, tmp_path / "t.json")
    assert list(t["rows"]) == sorted(ids)


def test_rho_matches_the_closed_form_and_the_section6_rules():
    rng = random.Random(3)
    u = {"moments": moments(rng), "identity_residual": 1e-7}
    c, m = u["moments"]["cov"], u["moments"]["mean"]
    for k in range(len(TERMS)):
        rho, vr, st = ex.rho_and_ratio(u, k)
        va, vt, cat = c[0][0], c[1 + k][1 + k], c[0][1 + k]
        beta = cat / va
        r2 = vt - cat ** 2 / va                                               # ||R_k||^2 per sample
        closed = (1 - beta) * math.sqrt(va) / math.sqrt((1 - beta) ** 2 * va + r2)
        assert st == "ok" and rho == pytest.approx(closed, rel=1e-9) and vr == pytest.approx(vt / va)
    # R_k = 0: T_k = beta*A exactly gives +1 or -1 by the sign of 1 - beta; beta = 1 makes A - T_k constant
    for beta, want in ((0.5, 1.0), (2.0, -1.0)):
        cov = [[1.0, beta], [beta, beta * beta]]
        r, _, st = ex.rho_and_ratio({"moments": {"mean": [0.0, 0.0], "cov": cov}, "identity_residual": 0.0}, 0)
        assert st == "ok" and r == pytest.approx(want)
    r, _, st = ex.rho_and_ratio({"moments": {"mean": [0, 0], "cov": [[1.0, 1.0], [1.0, 1.0]]}, "identity_residual": 0}, 0)
    assert st == "undefined" and r is None
    # a near-constant A, with a large mean, is caught (the old 1e-12 x Var(A) test could never fire)
    tiny = {"moments": {"mean": [50.0, 0.0], "cov": [[1e-12, 0.0], [0.0, 1.0]]}, "identity_residual": 0.0}
    assert ex.rho_and_ratio(tiny, 0)[2] == "undefined"
    # non-finite moments, a broken identity, and |rho| past 1 + 1e-6 are invalid
    assert ex.rho_and_ratio({"moments": {"mean": [0, 0], "cov": [[float("nan"), 0], [0, 1]]},
                             "identity_residual": 0}, 0)[2] == "invalid"
    assert ex.rho_and_ratio(dict(u, identity_residual=1.0), 0)[2] == "invalid"
    impossible = {"moments": {"mean": [0, 0], "cov": [[1.0, -0.5], [-0.5, 0.1]]}, "identity_residual": 0}
    assert ex.rho_and_ratio(impossible, 0)[2] == "invalid"                    # not a valid covariance matrix


def test_a_window_with_mostly_undefined_rho_is_missing(tmp_path):
    ev, y = make_robot(tmp_path, "a", BASE_W, iterations=(0, 1, 2))
    ups = [json.loads(x) for x in open(f"{ev}/updates.jsonl")]
    for u in ups[:2]:
        u["identity_residual"] = 5.0
    with open(f"{ev}/updates.jsonl", "w") as f:
        f.writelines(json.dumps(u) + "\n" for u in ups)
    f, counts = ex.export_robot(ev, y)
    assert f["C:rho:feet_slide:w1"] is None and f["Bvar:vr:feet_slide:w1"] is None
    assert f["B:unw:feet_slide:w1"] is not None and counts["rho_invalid"] == 2 * len(TERMS)
    assert f["C:rho:feet_slide:w2"] is None                                   # no update in window 2 at all


def test_cliff_index_uses_the_v2_grid():
    assert ex.cliff_index("action_latency_ms", None) == ex.NO_CLIFF
    assert ex.cliff_index("action_latency_ms", 25.0) == 5                    # 0, 5, ..., 45 ms
    assert ex.cliff_index("ground_friction", 0.8) == 0


def test_leak_audit_flags_a_configuration_value_and_reports_the_weight_clue(tmp_path):
    robots = []
    for arm, w in (("Control", BASE_W), ("R-slip", dict(BASE_W, feet_slide=-0.025))):
        for s in range(2):
            ev, y = make_robot(tmp_path, f"{arm}{s}", w, behaviour_seed=hash((arm, s)) % 1000)
            robots.append({"label": arm, "evidence": ev, "env_yaml": y})
    rep = ex.leak_audit(robots)
    assert rep["1_same_terms_order_format"]["pass"] and rep["pass"]
    assert rep["2_constant_within_arm_features"]["found"] == []
    assert rep["3_trivial_clue_accuracy"]["accuracy"] == 1.0                 # the weights give the arm away
    # plant a configuration value in a primary input: identical within arms, different between them
    for r in robots:
        ups = [json.loads(x) for x in open(f"{r['evidence']}/updates.jsonl")]
        for u in ups:
            u["learning_rate"] = 1e-3 if r["label"] == "Control" else 2e-3
        with open(f"{r['evidence']}/updates.jsonl", "w") as f:
            f.writelines(json.dumps(u) + "\n" for u in ups)
    found = ex.leak_audit(robots)["2_constant_within_arm_features"]["found"]
    assert found == [f"B:log:learning_rate:w{i}" for i in (1, 2, 3)]


def test_leak_audit_fails_on_different_term_lists(tmp_path):
    ev1, y1 = make_robot(tmp_path, "a", BASE_W)
    ev2, y2 = make_robot(tmp_path, "b", BASE_W)
    meta = json.load(open(f"{ev2}/meta.json"))
    meta["reward_terms"] = TERMS[::-1]
    json.dump(meta, open(f"{ev2}/meta.json", "w"))
    rep = ex.leak_audit([{"label": "x", "evidence": ev1, "env_yaml": y1}, {"label": "y", "evidence": ev2, "env_yaml": y2}])
    assert not rep["pass"] and not rep["1_same_terms_order_format"]["pass"]


def test_audit_command_reads_pilot_work_folders(tmp_path, monkeypatch, capsys):
    from sim2gate.v2 import phase2_arms as arms
    work = tmp_path / "p"
    robots = {}
    for task, (arm, cid) in list(arms.all_tasks().items())[:2]:
        for s in range(2):
            name = f"p2pilot_{arm}_{cid}_s{s}".replace(".", "_")
            ev, y = make_robot(work / "robots", name, BASE_W, behaviour_seed=s)
            run = tmp_path / "logs" / name
            (run / "params").mkdir(parents=True)
            (run / "params" / "env.yaml").write_text(open(y).read())
            robots[name] = {"finished": "t", "problems": [], "task": task, "run_dir": str(run)}
    robots["control_s101"] = {"finished": "t", "problems": [], "task": "Sim2Gate-Velocity-Flat-Go2-P0-v5b"}
    (work / "state.json").write_text(json.dumps({"robots": robots}))
    out = tmp_path / "audit.json"
    assert ex.main(["audit", "--work", str(work), "--out", str(out)]) == 0
    rep = json.load(open(out))
    assert rep["robots"] == 4 and "control_s101" not in rep["robot_names"]
    assert "leak audit: 4 pilot robots" in capsys.readouterr().out
