"""v2 phase 0B analysis, exactly as frozen in docs/v2/PHASE0B_PROTOCOL.md and PHASE0B_FREEZE.md (amendment 2).

Reads the fingerprinted phase 0A bundle, computes Q0-Q3, applies the frozen rules and writes a report. Needs only
numpy and torch (CPU is fine; a few minutes).

    python -m sim2gate.v2.phase0b --bundle phase0a-evidence.tgz --out phase0b_results

Prints the report; writes <out>/phase0b_report.md and <out>/phase0b_results.json.
The answer is provisional until Vihaan signs off Q1's derivation, the cross-implementation gradient check and Q4.
"""
import argparse
import copy
import glob
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile

import numpy as np

BUNDLE_SHA256 = "a893840106dfd70c42cd9b9aefdaa871f73605e9fb8fec0d6c822b678b929961"
PPO_RUNS = ("base_s1", "base_s2", "no_slip_s1", "no_slip_s2", "no_action_rate_s1", "no_action_rate_s2")
BSRS_RUNS = {"bsrs_0.5_s1": 0.5, "bsrs_0.5_s2": 0.5, "bsrs_2_s1": 2.0, "bsrs_2_s2": 2.0}
ETAS = ("0.5", "1", "2")
REFIT_RUN, REFIT_TERMS, REFIT_ITERS = "base_s1", ("feet_slide", "action_rate_l2"), (50, 200, 500, 1000, 1999)
WARMUP = 50
N_BOOT = 10_000
MARGIN = 0.1
Q0_MIN, Q0_LOW = 0.7, 0.5
REL_TOL = 1e-5
T2_MAX_BELOW = 2


# ---- statistics ----------------------------------------------------------------------------------------------------

def _rank(x):
    """Ranks with ties averaged (vectorized)."""
    x = np.asarray(x, float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts) - 1.0
    return (ends - (counts - 1) / 2.0)[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / den) if den > 0 else float("nan")


def bootstrap_runs(pairs_by_run, stat, n=N_BOOT, seed=0):
    """95% percentile interval of stat(list of per-run pair sets) with runs resampled with replacement."""
    rng = np.random.default_rng(seed)
    runs = list(pairs_by_run)
    vals = []
    for _ in range(n):
        pick = [runs[i] for i in rng.integers(0, len(runs), len(runs))]
        v = stat([pairs_by_run[r] for r in pick])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def three_way(low, high, threshold):
    """pass if low > threshold, fail if high < threshold, else inconclusive."""
    if not (np.isfinite(low) and np.isfinite(high)):
        return "inconclusive"
    return "pass" if low > threshold else "fail" if high < threshold else "inconclusive"


# ---- reading -------------------------------------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def open_bundle(bundle, workdir):
    if os.path.isdir(bundle):
        root = bundle
    else:
        digest = sha256(bundle)
        if digest != BUNDLE_SHA256:
            raise SystemExit(f"bundle SHA-256 {digest} is not the phase 0A bundle ({BUNDLE_SHA256})")
        with tarfile.open(bundle) as t:
            safe = [m for m in t.getmembers() if not (m.name.startswith("/") or ".." in m.name.split("/"))]
            t.extractall(workdir, members=safe)
        root = workdir
    cand = glob.glob(os.path.join(root, "**", "runs"), recursive=True)
    if not cand:
        raise SystemExit("no runs/ folder in the bundle")
    return cand[0]


def load_run(runs_dir, name):
    d = os.path.join(runs_dir, name, "evidence")
    meta = json.load(open(os.path.join(d, "meta.json")))
    recs = [json.loads(x) for x in open(os.path.join(d, "updates.jsonl")) if x.strip()]
    return {"name": name, "dir": d, "meta": meta, "recs": recs}


def moments_baselines(rec, k):
    """Var(T_k)/Var(A) and beta_k from the recorded covariance of [A, T_1..T_K, V]."""
    cov = np.asarray(rec["moments"]["cov"], float)
    va = cov[0, 0]
    if va <= 0:
        return float("nan"), float("nan")
    return cov[1 + k, 1 + k] / va, cov[0, 1 + k] / va


# ---- Q0, Q2, T1 ----------------------------------------------------------------------------------------------------

def q0_pairs(run, min_it=0):
    out = []
    for r in run["recs"]:
        if not r.get("deep") or r["iteration"] < min_it:
            continue
        for e in ETAS:
            rho = (r.get("bsrs_what_if_rho") or {}).get(e)
            cos = (r.get("direction_cos_trainer") or {}).get(f"bsrs:{e}")
            if rho is not None and cos is not None:
                out.append((rho, cos))
    return out


def q2_rows(run, min_it=0):
    """(rho_k, cos_k, abs_share, var_ratio, beta_k) per deep iteration and term."""
    names = run["meta"]["reward_terms"]
    out = []
    for r in run["recs"]:
        if not r.get("deep") or r["iteration"] < min_it:
            continue
        dc = r.get("direction_cos_trainer") or {}
        share = r.get("term_abs_share") or [float("nan")] * len(names)
        for k, n in enumerate(names):
            rho, cos = r["corr_A_A_minus_term"][k], dc.get(f"minus:{n}")
            if rho is None or cos is None:
                continue
            vr, beta = moments_baselines(r, k)
            out.append((rho, cos, share[k], vr, beta))
    return out


def _sp_pooled(sets, i, j):
    rows = [p for s in sets for p in s]
    if not rows:
        return float("nan")
    a = np.asarray(rows, float)
    return spearman(a[:, i], a[:, j])


def analyse_t1(runs, min_it):
    q0 = {r["name"]: q0_pairs(r, min_it) for r in runs}
    q2 = {r["name"]: q2_rows(r, min_it) for r in runs}
    s_ref = _sp_pooled(list(q0.values()), 0, 1)
    s_k = _sp_pooled(list(q2.values()), 0, 1)
    both = {n: (q0[n], q2[n]) for n in q0}

    def diff(sets):
        return _sp_pooled([s[1] for s in sets], 0, 1) - _sp_pooled([s[0] for s in sets], 0, 1)
    q0_ci = bootstrap_runs(q0, lambda sets: _sp_pooled(sets, 0, 1))
    d_ci = bootstrap_runs(both, diff)
    q0_ok = (s_ref >= Q0_MIN) and (q0_ci[0] > Q0_LOW)
    t1 = three_way(d_ci[0], d_ci[1], -MARGIN)
    baselines = {}
    for col, label in ((2, "term_abs_share"), (3, "var_T_k_over_var_A"), (4, "beta_k")):
        s_b = _sp_pooled(list(q2.values()), col, 1)

        def bdiff(sets, col=col):
            return abs(_sp_pooled(sets, 0, 1)) - abs(_sp_pooled(sets, col, 1))
        ci = bootstrap_runs(q2, bdiff)
        baselines[label] = {"spearman": s_b, "abs_diff": abs(s_k) - abs(s_b), "ci95": ci,
                            "verdict": three_way(ci[0], ci[1], 0.0)}
    bverd = [b["verdict"] for b in baselines.values()]
    base_verdict = "fail" if "fail" in bverd else "pass" if all(v == "pass" for v in bverd) else "inconclusive"
    return {"min_iteration": min_it, "n_pairs_q0": sum(len(v) for v in q0.values()),
            "n_pairs_q2": sum(len(v) for v in q2.values()),
            "q0": {"spearman": s_ref, "ci95": q0_ci, "pass": bool(q0_ok)},
            "q2": {"spearman": s_k},
            "t1": {"diff": s_k - s_ref, "ci95": d_ci, "verdict": t1},
            "baselines": baselines, "baselines_verdict": base_verdict}


def bsrs_descriptive(runs):
    out = {}
    for r in runs:
        eta = BSRS_RUNS[r["name"]]
        key = f"{eta:g}"
        pairs = [(x["bsrs_what_if_rho"][key], x["direction_cos_trainer"]["A"]) for x in r["recs"]
                 if x.get("deep") and x.get("direction_cos_trainer", {}).get("A") is not None]
        a = np.asarray(pairs, float)
        out[r["name"]] = {"eta": eta, "n": len(pairs), "rho_mean": float(a[:, 0].mean()),
                          "cos_A_mean": float(a[:, 1].mean()), "spearman": spearman(a[:, 0], a[:, 1])}
    return out


# ---- Q3, T2 --------------------------------------------------------------------------------------------------------

def _cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else float("nan")


def analyse_t2(run):
    recs = {r["iteration"]: r for r in run["recs"]}
    points = []
    for it in REFIT_ITERS:
        f = os.path.join(run["dir"], "deep", f"it{it:05d}_directions.npz")
        if not os.path.exists(f) or it not in recs:
            for term in REFIT_TERMS:
                points.append({"iteration": it, "term": term, "missing": True})
            continue
        z = np.load(f)
        names = [str(n) for n in z["names"]]
        D = z["directions"]
        yard = float(np.median(recs[it].get("minibatch_grad_cos_trainer") or [np.nan]))
        for term in REFIT_TERMS:
            try:
                c = _cos(D[names.index(f"refit_minus:{term}")], D[names.index(f"minus:{term}")])
            except ValueError:
                points.append({"iteration": it, "term": term, "missing": True})
                continue
            rf = (recs[it].get("refit") or {}).get(term, {})
            points.append({"iteration": it, "term": term, "cos_refit_vs_same_critic": c, "yardstick": yard,
                           "below": bool(c < yard), "corr_A_minus_term_vs_refit": rf.get("corr_A_minus_term"),
                           "refit_value_loss_by_round": rf.get("value_loss_by_round")})
    below = sum(p.get("below", False) for p in points)
    missing = sum(p.get("missing", False) for p in points)
    verdict = "inconclusive" if missing else ("fail" if below > T2_MAX_BELOW else "pass")
    return {"points": points, "below": below, "missing": missing, "verdict": verdict}


# ---- Q1 numerics and the offline null refit (torch) ---------------------------------------------------------------

def _mlp(sd, prefix, torch):
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith(prefix + ".") and k.endswith(".weight")})
    layers = []
    for n, i in enumerate(idx):
        w, b = sd[f"{prefix}.{i}.weight"], sd[f"{prefix}.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w); lin.bias.data.copy_(b)
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).double()


def _normalizer(sd, torch):
    if any(k.startswith("obs_normalizer.") for k in sd):
        raise NotImplementedError("observation normalization is on; the phase 0A task has it off")
    return lambda x: x


def _std(sd, torch):
    if "distribution.std_param" in sd:
        return torch.nn.Parameter(sd["distribution.std_param"].double().clone())
    if "distribution.log_std_param" in sd:
        return torch.nn.Parameter(sd["distribution.log_std_param"].double().clone())
    raise KeyError("no std parameter in the actor state_dict")


def _load_sample(run, it, torch):
    z = np.load(os.path.join(run["dir"], "deep", f"it{it:05d}_samples.npz"))
    t = lambda k: torch.from_numpy(z[k]).double()  # noqa: E731
    obs = t("obs_policy")
    return z, t, obs


def q1_numerics(runs, torch):
    from ..training.signals import gae, reward_tail, shaped_rewards
    worst = {"A_vs_trainer": 0.0, "terms_vs_reward": 0.0, "closed_form_rho_k": 0.0}
    n = 0
    for run in runs:
        m = run["meta"]
        g, lam, eta = m["gamma"], m["lam"], float(m.get("eta") or 0.0)
        for f in sorted(glob.glob(os.path.join(run["dir"], "deep", "it*_samples.npz"))):
            z = np.load(f)
            t = lambda k: torch.from_numpy(z[k]).double()  # noqa: E731
            d, to, v, lv, r = t("dones").bool(), t("time_outs").bool(), t("values"), t("last_values"), t("raw_rewards")
            A = gae(r, v, lv, d, to, gamma=g, lam=lam)
            tr = A if eta == 0 else gae(shaped_rewards(r, v, lv, d, to, eta=eta, gamma=g), v, lv, d, to,
                                        gamma=g, lam=lam)
            ref = t("trainer_advantages_unnormalized")
            worst["A_vs_trainer"] = max(worst["A_vs_trainer"], float((tr - ref).abs().max() / ref.abs().max()))
            terms = t("terms")
            worst["terms_vs_reward"] = max(worst["terms_vs_reward"],
                                           float((terms.sum(-1) - r).abs().max() / r.abs().max()))
            a = A.reshape(-1)
            ac = a - a.mean()
            for k in range(terms.shape[-1]):
                Tk = reward_tail(terms[..., k], d, gamma=g, lam=lam).reshape(-1)
                tc = Tk - Tk.mean()
                beta = float(tc @ ac / (ac @ ac))
                R = tc - beta * ac
                closed = (1 - beta) * ac.norm() / (((1 - beta) * ac.norm()) ** 2 + R.norm() ** 2).sqrt()
                y = ac - tc
                direct = float(ac @ y / (ac.norm() * y.norm()))
                worst["closed_form_rho_k"] = max(worst["closed_form_rho_k"], abs(float(closed) - direct))
            n += 1
    return {"sample_dumps": n, "max_relative_error": worst,
            "pass": all(v <= REL_TOL for v in worst.values())}


def null_refit(run, torch, seed=0):
    from ..training.signals import gae
    m = run["meta"]
    g, lam = m["gamma"], m["lam"]
    names = m["reward_terms"]
    nmb = int(m["num_mini_batches"])
    out = []
    for it in REFIT_ITERS:
        mp = os.path.join(run["dir"], "deep", f"it{it:05d}_models.pt")
        if not os.path.exists(mp):
            out.append({"iteration": it, "missing": True})
            continue
        torch.manual_seed(seed)
        sd = torch.load(mp, map_location="cpu")
        z, t, obs = _load_sample(run, it, torch)
        T, S = obs.shape[:2]
        a_sd, c_sd = sd["actor"], sd["critic"]
        a_norm, c_norm = _normalizer(a_sd, torch), _normalizer(c_sd, torch)
        actor, std_p = _mlp(a_sd, "mlp", torch), _std(a_sd, torch)
        log_std = "distribution.log_std_param" in a_sd
        crit0 = _mlp(c_sd, "mlp", torch)
        x = obs.reshape(T * S, -1)
        acts = t("actions").reshape(T * S, -1)
        d, to, lv = t("dones").bool(), t("time_outs").bool(), t("last_values")
        v_rec, raw, terms = t("values"), t("raw_rewards"), t("terms")
        params = list(actor.parameters()) + [std_p]

        def direction(adv):
            a = adv.reshape(-1)
            a = (a - a.mean()) / (a.std() + 1e-8)
            mean = actor(a_norm(x))
            std = std_p.exp() if log_std else std_p.clamp(1e-6, 1e6)
            logp = torch.distributions.Normal(mean, std).log_prob(acts).sum(-1)
            gr = torch.autograd.grad((a.detach() * logp).mean(), params)
            return torch.cat([q.reshape(-1) for q in gr]).detach().numpy()

        def refit(r):
            crit = copy.deepcopy(crit0)
            opt = torch.optim.Adam(crit.parameters(), lr=1e-3)
            n = T * S
            mb = max(n // nmb, 1)
            for _ in range(3):
                with torch.no_grad():
                    v = crit(c_norm(x)).reshape(T, S)
                    target = (gae(r, v, lv, d, to, gamma=g, lam=lam) + v).reshape(-1)
                for _e in range(5):
                    perm = torch.randperm(n)
                    for i in range(0, n - mb + 1, mb):
                        ix = perm[i:i + mb]
                        loss = (crit(c_norm(x[ix])).reshape(-1) - target[ix]).pow(2).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = crit(c_norm(x)).reshape(T, S)
            return gae(r, v, lv, d, to, gamma=g, lam=lam)

        row = {"iteration": it}
        for label, r in (("null", raw),) + tuple((f"minus:{n}", raw - terms[..., names.index(n)])
                                                  for n in REFIT_TERMS):
            same = gae(r, v_rec, lv, d, to, gamma=g, lam=lam)
            row[label] = _cos(direction(refit(r)), direction(same))
        out.append(row)
    return out


# ---- decision and report -------------------------------------------------------------------------------------------

def decide(main, warm, t2, q1):
    def answer(t1res):
        if not t1res["q0"]["pass"]:
            return "Q0 fails"
        if not q1["pass"]:
            return "inconclusive (Q1 numerics)"
        v, b = t1res["t1"]["verdict"], t1res["baselines_verdict"]
        if v == "fail" or b == "fail":
            return "(c) stays with shaping terms"
        if v == "inconclusive" or b == "inconclusive":
            return "inconclusive"
        if t2["verdict"] == "fail":
            return "(b) needs a new derivation"
        if t2["verdict"] == "inconclusive":
            return "inconclusive"
        return "(a) extends as it stands"
    a_main, a_warm = answer(main), answer(warm)
    return {"main": a_main, "without_warmup": a_warm,
            "answer": a_main if a_main == a_warm else "inconclusive (warm-up check disagrees)"}


def _f(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def report(res):
    m, w, t2, q1, dec = res["t1_main"], res["t1_without_warmup"], res["t2"], res["q1"], res["decision"]
    L = ["# Phase 0B results (provisional until Vihaan signs off Q1 and Q4)", "",
         f"Bundle SHA-256 {res['bundle_sha256']}. Rules: docs/v2/PHASE0B_PROTOCOL.md (frozen) and amendment 2.", "",
         f"## Answer under the frozen rules: **{dec['answer']}**", "",
         f"- All deep iterations: {dec['main']}", f"- Without iterations 0-{WARMUP - 1}: {dec['without_warmup']}", "",
         "## Q0, BSRS reference (6 PPO runs)", "",
         f"Spearman(rho, cos) = {_f(m['q0']['spearman'])}, 95% CI {_f(m['q0']['ci95'][0])} to "
         f"{_f(m['q0']['ci95'][1])}, {m['n_pairs_q0']} pairs. Rule: >= {Q0_MIN} and lower bound > {Q0_LOW}: "
         f"**{'pass' if m['q0']['pass'] else 'fail'}**.", "",
         "BSRS arms, descriptive (training-eta rho vs cos with g(A)):", ""]
    for n, b in res["bsrs_descriptive"].items():
        L.append(f"- {n}: eta {b['eta']:g}, mean rho {_f(b['rho_mean'])}, mean cos {_f(b['cos_A_mean'])}, "
                 f"Spearman {_f(b['spearman'])} (n={b['n']})")
    L += ["", "## Q1, numerics on the sample dumps", "",
          f"{q1['sample_dumps']} dumps. Max relative error: "
          + ", ".join(f"{k} {v:.1e}" for k, v in q1["max_relative_error"].items())
          + f". Rule <= {REL_TOL:g}: **{'pass' if q1['pass'] else 'fail'}**. The derivation and the "
          "cross-implementation gradient check are Vihaan's.", "",
          "## Q2 and T1, per-term rho against the reference", ""]
    for lab, r in (("All deep iterations", m), (f"Without 0-{WARMUP - 1}", w)):
        L.append(f"- {lab}: Spearman per-term {_f(r['q2']['spearman'])} vs reference {_f(r['q0']['spearman'])}; "
                 f"difference {_f(r['t1']['diff'])}, 95% CI {_f(r['t1']['ci95'][0])} to {_f(r['t1']['ci95'][1])} "
                 f"(margin -{MARGIN}): **{r['t1']['verdict']}** ({r['n_pairs_q2']} pairs)")
    L += ["", "Baselines (|S_rho| - |S_baseline|, all deep iterations):", ""]
    for n, b in m["baselines"].items():
        L.append(f"- {n}: Spearman {_f(b['spearman'])}, difference {_f(b['abs_diff'])}, 95% CI "
                 f"{_f(b['ci95'][0])} to {_f(b['ci95'][1])}: {b['verdict']}")
    L += ["", f"## Q3 and T2, critic refits ({REFIT_RUN})", "",
          "| Iteration | Term | cos(refit, same critic) | Yardstick (median minibatch cos) | Below |",
          "|---|---|---|---|---|"]
    for p in t2["points"]:
        if p.get("missing"):
            L.append(f"| {p['iteration']} | {p['term']} | missing | | |")
        else:
            L.append(f"| {p['iteration']} | {p['term']} | {_f(p['cos_refit_vs_same_critic'])} | "
                     f"{_f(p['yardstick'])} | {'yes' if p['below'] else 'no'} |")
    L += ["", f"{t2['below']} of 10 below the yardstick (fails if more than {T2_MAX_BELOW}): **{t2['verdict']}**.", "",
          "Offline null refit (context only; cos of refitted vs same-critic direction on the 256-env sample):", ""]
    for r in res["null_refit"]:
        L.append(f"- it {r['iteration']}: " + ", ".join(f"{k} {_f(v)}" for k, v in r.items() if k != "iteration"))
    L += ["", "## Not decided here", "",
          "- Q4: what per-term rho may and may not be claimed to mean (Vihaan).",
          "- Whether per-term rho predicts behaviour in the finished robot: phase 2, not phase 0B."]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.v2.phase0b", description=__doc__.splitlines()[0])
    p.add_argument("--bundle", required=True, help="phase0a-evidence.tgz (or an already extracted folder)")
    p.add_argument("--out", default="phase0b_results")
    p.add_argument("--skip-torch", action="store_true", help="skip Q1 numerics and the null refit (tests only)")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = open_bundle(a.bundle, tmp)
        ppo = [load_run(runs_dir, n) for n in PPO_RUNS]
        bsrs = [load_run(runs_dir, n) for n in BSRS_RUNS]
        refit_run = next(r for r in ppo if r["name"] == REFIT_RUN)
        res = {"bundle_sha256": BUNDLE_SHA256 if not os.path.isdir(a.bundle) else "folder (not verified)",
               "t1_main": analyse_t1(ppo, 0), "t1_without_warmup": analyse_t1(ppo, WARMUP),
               "bsrs_descriptive": bsrs_descriptive(bsrs), "t2": analyse_t2(refit_run)}
        if a.skip_torch:
            res["q1"] = {"sample_dumps": 0, "max_relative_error": {}, "pass": True, "skipped": True}
            res["null_refit"] = []
        else:
            import torch
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            res["q1"] = q1_numerics(ppo + bsrs, torch)
            res["null_refit"] = null_refit(refit_run, torch)
        res["decision"] = decide(res["t1_main"], res["t1_without_warmup"], res["t2"], res["q1"])
    text = report(res)
    with open(os.path.join(a.out, "phase0b_report.md"), "w") as f:
        f.write(text)
    with open(os.path.join(a.out, "phase0b_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
