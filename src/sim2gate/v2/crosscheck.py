"""Phase 0B Q1's gradient cross-check (PHASE0B_PROTOCOL.md 4.5; PHASE2_PROTOCOL.md "Review corrections").

Two implementations compute, on the same 256-environment sample and from the same saved actor (`models.pt`), the
direction g(A_hat) = grad_theta mean(A_hat * log pi_theta(a|s)) for each variant below. They pass if every pair of
directions has cosine >= 0.999999.

This module is Sim2Gate's side. It reads only the phase 0A bundle (format: PHASE0A_FORMAT.md) and builds the
Gaussian MLP policy directly from the state dict, so it does not use the recorder's code or RSL-RL's classes. It
checks that the rebuilt actor is the saved one before writing anything (see the identity checks below).

Variants, over all T x 256 samples of one dump (float64 throughout):
- `trainer`: A_hat = the sample's `trainer_advantages`, as recorded (normalised with full-rollout statistics);
- `A`: GAE of `raw_rewards` with `values`, `last_values`, `dones`, `time_outs` (FORMAT.md's recursion), then
  normalised over the sample: (x - mean) / (std + 1e-8), std with n - 1;
- `minus:<term>`: A - T_k, where T_k is GAE of term k's rewards with V = 0, normalised the same way.

Output: one `<run>_it<NNNNN>.npz` per run and grad-save iteration, with `names` [M], `param_names` [n],
`param_shapes`, `directions` [M, P] (float64; parameters flattened in state-dict order), and `MANIFEST.json` with
every file's SHA-256.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

COS_MIN = 0.999999
EPS = 1e-8


def gae(r, v, last_v, dones, time_outs, gamma, lam):
    """FORMAT.md: delta_t = r_t + gamma*next_t - V_t; next_t = V_{t+1} (last_values at T-1), V_t on a time-out,
    0 on a termination; A_t = delta_t + gamma*lam*(1 - done_t)*A_{t+1}."""
    import numpy as np
    T = r.shape[0]
    out = np.zeros_like(r)
    nxt_adv = np.zeros_like(r[0])
    for t in range(T - 1, -1, -1):
        nv = last_v if t == T - 1 else v[t + 1]
        term = (dones[t] > 0) & ~(time_outs[t] > 0)
        nxt = np.where(time_outs[t] > 0, v[t], np.where(term, 0.0, nv))
        delta = r[t] + gamma * nxt - v[t]
        nxt_adv = delta + gamma * lam * (1.0 - (dones[t] > 0)) * nxt_adv
        out[t] = nxt_adv
    return out


def normalise(x):
    return (x - x.mean()) / (x.std(ddof=1) + EPS)


def variants(s, gamma, lam):
    import numpy as np
    f = {k: np.asarray(s[k], dtype=np.float64) for k in ("raw_rewards", "values", "last_values", "dones", "time_outs")}
    A = gae(f["raw_rewards"], f["values"], f["last_values"], f["dones"], f["time_outs"], gamma, lam)
    terms = np.asarray(s["terms"], dtype=np.float64)
    zero_v, zero_last = np.zeros_like(f["values"]), np.zeros_like(f["last_values"])
    out = {"trainer": np.asarray(s["trainer_advantages"], dtype=np.float64), "A": normalise(A)}
    for k, n in enumerate([str(x) for x in s["term_names"]]):
        Tk = gae(terms[..., k], zero_v, zero_last, f["dones"], f["time_outs"], gamma, lam)
        out[f"minus:{n}"] = normalise(A - Tk)
    return out


class GaussianMLP:
    """The policy from a state dict: Linear layers `...<i>.weight/bias` with ELU between them, and a std parameter."""

    def __init__(self, sd):
        import torch
        self.names = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()
                      and (k.endswith(".weight") or k.endswith(".bias") or "std" in k)]
        self.params = {k: sd[k].detach().double().clone().requires_grad_(True) for k in self.names}
        lin = sorted({k.rsplit(".", 1)[0] for k in self.names if k.endswith(".weight")},
                     key=lambda p: int(p.rsplit(".", 1)[1]))
        self.layers = [(self.params[p + ".weight"], self.params[p + ".bias"]) for p in lin]
        stds = [k for k in self.names if "std" in k]
        if len(stds) != 1:
            raise ValueError(f"expected one std parameter, found {stds}")
        self.std_key = stds[0]

    def dist(self, obs):
        import torch
        x = obs
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = torch.nn.functional.elu(x)
        p = self.params[self.std_key]
        std = p.exp() if "log" in self.std_key else p
        return x, std.expand_as(x)

    def log_prob(self, obs, actions):
        import math
        mean, std = self.dist(obs)
        return (-((actions - mean) ** 2) / (2 * std ** 2) - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)


# Identity checks of the rebuilt policy (revised Oct 25, 2026, after Vihaan's run report and before Sim2Gate's
# reference was computed). The rollout was collected on the GPU, where matrix products run at reduced (TF32)
# precision, so the saved weights' float64 forward pass reproduces the recorded Gaussian mean only to about 1e-3,
# and log pi to about 1e-2. The checks therefore establish that the rebuilt actor is the saved actor, not
# bit-equality: the std must match exactly, the recorded mean and std must reproduce the recorded log pi (the
# density code), and the mean must match to MEAN_TOL. Reported as well: whether the saved actor fits the recorded mean
# at least NEIGHBOUR_FACTOR times better than the actor of the nearest other grad-save iteration. The pass criterion of the
# cross-check itself (cosine >= COS_MIN between the two implementations) is unchanged.
STD_TOL = 1e-6
DENSITY_TOL = 1e-4
MEAN_TOL = 2e-2
NEIGHBOUR_FACTOR = 10.0


def _gauss_logp(actions, mean, std):
    import math
    return (-((actions - mean) ** 2) / (2 * std ** 2) - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)


def directions(models_pt, samples_npz, gamma, lam, neighbour_models_pt=None):
    import numpy as np
    import torch
    s = np.load(samples_npz, allow_pickle=False)
    actor = GaussianMLP(torch.load(models_pt, map_location="cpu", weights_only=True)["actor"])
    obs_keys = [k for k in s.files if k.startswith("obs_")]
    if len(obs_keys) != 1:
        raise ValueError(f"expected one observation group, found {obs_keys}")
    obs = torch.from_numpy(np.asarray(s[obs_keys[0]], dtype=np.float64)).flatten(0, 1)
    act = torch.from_numpy(np.asarray(s["actions"], dtype=np.float64)).flatten(0, 1)
    logp = actor.log_prob(obs, act)
    with torch.no_grad():
        mean, std = actor.dist(obs)
        rec_mean = torch.from_numpy(np.asarray(s["old_dist_param_0"], np.float64)).flatten(0, 1)
        rec_std = torch.from_numpy(np.asarray(s["old_dist_param_1"], np.float64)).flatten(0, 1)
        rec_logp = torch.from_numpy(np.asarray(s["old_log_prob"], dtype=np.float64)).flatten()
        checks = {
            "log_prob_max_abs_diff": float((logp - rec_logp).abs().max()),
            "mean_max_abs_diff": float((mean - rec_mean).abs().max()),
            "std_max_abs_diff": float((std - rec_std).abs().max()),
            "density_from_recorded_max_abs_diff": float((_gauss_logp(act, rec_mean, rec_std) - rec_logp).abs().max())}
        if neighbour_models_pt:
            other = GaussianMLP(torch.load(neighbour_models_pt, map_location="cpu", weights_only=True)["actor"])
            checks["neighbour_mean_max_abs_diff"] = float((other.dist(obs)[0] - rec_mean).abs().max())
    bad = []
    if checks["std_max_abs_diff"] > STD_TOL:
        bad.append("std differs")
    if checks["density_from_recorded_max_abs_diff"] > DENSITY_TOL:
        bad.append("recorded mean and std do not reproduce the recorded log-probability")
    if checks["mean_max_abs_diff"] > MEAN_TOL:
        bad.append("mean differs beyond collection precision")
    if "neighbour_mean_max_abs_diff" in checks:                               # reported, not a stop
        checks["neighbour_separated"] = bool(
            checks["neighbour_mean_max_abs_diff"] >= NEIGHBOUR_FACTOR * checks["mean_max_abs_diff"])
    if bad:
        raise ValueError(f"the rebuilt policy is not the saved one ({'; '.join(bad)}): {checks}")
    vs = variants(s, gamma, lam)
    params = [actor.params[k] for k in actor.names]
    out = []
    for i, (name, a) in enumerate(vs.items()):
        a_hat = torch.from_numpy(a).flatten()
        g = torch.autograd.grad((a_hat * logp).mean(), params, retain_graph=i < len(vs) - 1)
        out.append(torch.cat([x.reshape(-1) for x in g]).numpy())
    return list(vs), actor.names, [list(actor.params[k].shape) for k in actor.names], np.stack(out), checks


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reference(bundle_dir, out_dir):
    """Sim2Gate's directions for every run at its grad-save iterations."""
    import numpy as np
    if not os.path.isdir(os.path.join(bundle_dir, "runs")):                 # an unpacked tgz: one folder deeper
        inner = [d for d in glob.glob(os.path.join(bundle_dir, "*")) if os.path.isdir(os.path.join(d, "runs"))]
        if len(inner) != 1:
            raise ValueError(f"no runs/ folder in {bundle_dir}")
        bundle_dir = inner[0]
    os.makedirs(out_dir, exist_ok=True)
    files, checks = {}, {}
    for run in sorted(os.listdir(os.path.join(bundle_dir, "runs"))):
        ev = os.path.join(bundle_dir, "runs", run, "evidence")
        meta = json.load(open(os.path.join(ev, "meta.json")))
        for it in meta["grad_save_iterations"]:
            sp = os.path.join(ev, "deep", f"it{it:05d}_samples.npz")
            mp = os.path.join(ev, "deep", f"it{it:05d}_models.pt")
            if not (os.path.exists(sp) and os.path.exists(mp)):
                continue
            deep = [i for i in meta["grad_save_iterations"] if i != it and
                    os.path.exists(os.path.join(ev, "deep", f"it{i:05d}_models.pt"))]
            nb = os.path.join(ev, "deep", f"it{min(deep, key=lambda i: abs(i - it)):05d}_models.pt") if deep else None
            names, pn, shapes, d, chk = directions(mp, sp, meta["gamma"], meta["lam"], nb)
            fn = f"{run}_it{it:05d}.npz"
            np.savez(os.path.join(out_dir, fn), names=np.array(names), param_names=np.array(pn),
                     param_shapes=np.array(json.dumps(shapes)), directions=d)
            files[fn] = sha256(os.path.join(out_dir, fn))
            checks[fn] = chk
    man = {"implementation": "sim2gate.v2.crosscheck", "files": files, "self_checks": checks}
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def compare(dir_a, dir_b):
    """Cosine of every matching direction; pass if all >= COS_MIN with identical file, variant and parameter lists."""
    import numpy as np
    fa = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_a, "*.npz")))
    fb = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dir_b, "*.npz")))
    problems, cosines = [], {}
    if fa != fb:
        problems.append(f"different files: only in A {sorted(set(fa) - set(fb))[:5]}, only in B "
                        f"{sorted(set(fb) - set(fa))[:5]}")
    for fn in sorted(set(fa) & set(fb)):
        a, b = np.load(os.path.join(dir_a, fn)), np.load(os.path.join(dir_b, fn))
        if [str(x) for x in a["param_names"]] != [str(x) for x in b["param_names"]]:
            problems.append(f"{fn}: parameter order differs")
            continue
        nb = {str(n): i for i, n in enumerate(b["names"])}
        for i, n in enumerate(str(x) for x in a["names"]):
            if n not in nb:
                problems.append(f"{fn}: variant {n} missing in B")
                continue
            x, y = a["directions"][i].astype(np.float64), b["directions"][nb[n]].astype(np.float64)
            den = float(np.linalg.norm(x) * np.linalg.norm(y))
            c = float(x @ y / den) if den > 0 else float("nan")
            cosines[f"{fn}:{n}"] = c
    vals = list(cosines.values())
    worst = min(vals) if vals else None
    fails = {k: v for k, v in cosines.items() if not (v >= COS_MIN)}
    return {"rule": f"cosine >= {COS_MIN} for every direction", "directions_compared": len(vals),
            "min_cosine": worst, "below_rule": dict(sorted(fails.items(), key=lambda kv: kv[1])[:20]),
            "problems": problems, "pass": bool(vals) and not fails and not problems}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.v2.crosscheck")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference", help="Sim2Gate's directions from the phase 0A bundle")
    r.add_argument("--bundle", required=True, help="the unpacked phase 0A bundle (the folder holding runs/)")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="compare two implementations' output folders")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out")
    a = p.parse_args(argv)
    import numpy as np
    if a.cmd == "reference":
        man = reference(a.bundle, a.out)
        sc = man["self_checks"].values()
        worst = {k: max((v.get(k, 0.0) for v in sc), default=None) for k in
                 ("mean_max_abs_diff", "log_prob_max_abs_diff", "std_max_abs_diff")}
        nbr = sum(bool(v.get("neighbour_separated")) for v in sc)
        n_dir = sum(len(np.load(os.path.join(a.out, f))["names"]) for f in man["files"])
        print(f"[sim2gate] cross-check reference: {len(man['files'])} files, {n_dir} directions; largest differences "
              f"from the recorded policy {worst}; neighbouring actor clearly worse in {nbr} of {len(man['files'])}; "
              f"manifest sha256 {sha256(os.path.join(a.out, 'MANIFEST.json'))}", flush=True)
        return 0
    rep = compare(a.a, a.b)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    print(f"[sim2gate] cross-check: {rep['directions_compared']} directions, min cosine {rep['min_cosine']}, "
          f"{'PASS' if rep['pass'] else 'FAIL'}", flush=True)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
