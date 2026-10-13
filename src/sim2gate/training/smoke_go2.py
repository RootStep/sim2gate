"""Experiment T step 1 in one command: the Go2 smoke test with automatic pass/fail.

Run inside an Isaac Lab 3.x environment, e.g. RootStep's Brev launchable:

    cd /workspaces/isaaclab_arena
    python -m sim2gate.training.smoke_go2            # --iterations 30 --seed 1 by default

It trains Isaac Lab's Go2 flat-ground task with identical settings: stock Isaac Lab (timing baseline), Sim2Gate
attached to stock PPO (--eta 0), Sim2Gate with BSRS (--eta 1), and a negative control that claims eta 1 but trains
without it, which the trainer-match check must catch. Then it checks the pass
conditions from docs/EXPERIMENT_T_RUNBOOK.md and writes report.json next to the logs. Exit code 0 = PASS.

--identity adds a second stock run with the same seed and compares final checkpoints (every tensor, optimizer
state included): the eta = 0 run with Sim2Gate attached must be bit-identical to stock Isaac Lab. The repeat
shows whether the setup is reproducible at all; if two stock runs already differ, identity cannot be shown by
checkpoint comparison and the check says so instead of passing.
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

ITER_RE = re.compile(r"Iteration time:\s*([0-9.]+)\s*s")
DEFAULT_TASK = "Isaac-Velocity-Flat-UnitreeGo2"


def isaac_python():
    p = os.environ.get("ISAAC_PYTHON")
    if p:
        return [p]
    if os.path.exists("/isaac-sim/python.sh"):
        return ["/isaac-sim/python.sh"]
    return [sys.executable]


def train_script():
    root = os.environ.get("ISAACLAB_PATH") or os.path.join(os.getcwd(), "submodules", "IsaacLab")
    path = os.path.join(root, "scripts", "reinforcement_learning", "train.py")
    if not os.path.exists(path):
        raise SystemExit(f"Isaac Lab train.py not found at {path}; set ISAACLAB_PATH to the Isaac Lab folder")
    return path


def run(cmd, log_path):
    """Run cmd, streaming output to the console and to log_path. Returns (returncode, text)."""
    print("\n$ " + " ".join(cmd), flush=True)
    lines = []
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            lines.append(line)
        proc.wait()
    return proc.returncode, "".join(lines)


def iteration_times(text, skip=2):
    times = [float(x) for x in ITER_RE.findall(text)]
    return times[skip:] if len(times) > skip else times


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


REQUIRED_RUNS = ("stock", "eta0", "eta1", "control")
TRAINER_TOL = 1e-4          # reconstruction vs the trainer's own float32 tensors, relative to their scale
CONTROL_MIN = 1e-2          # the negative control must miss by at least this much


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def checks(iterations, eta0, eta1, returncodes, eta=1.0, control=None):
    """Pass conditions from the runbook. Returns a list of {name, ok, detail}. Every number a check relies on
    must be present and finite; a missing run or record fails the check rather than passing it."""
    out = []
    add = lambda name, ok, detail: out.append({"name": name, "ok": bool(ok), "detail": detail})
    need = [k for k in REQUIRED_RUNS if k != "control" or control is not None]
    add("all runs exited cleanly", all(k in returncodes for k in need) and all(v == 0 for v in returncodes.values()),
        returncodes)
    add("one record per iteration", len(eta0) == iterations and len(eta1) == iterations,
        {"eta0": len(eta0), "eta1": len(eta1), "expected": iterations})
    recs = eta0 + eta1
    if not recs:
        add("signals recorded", False, "no sim2gate_signals records")
        return out
    S = [r["signals"] for r in recs]
    sizes = {s.get("num_samples") for s in S}
    add("samples per iteration constant", len(sizes) == 1 and all(isinstance(x, int) and x > 0 for x in sizes),
        sorted(sizes, key=str))
    ident = [s.get("identity_residual") for s in S]
    add("BSRS identity holds (float32 rounding)", all(_fin(x) and 0 <= x <= 1e-3 for x in ident),
        max((x for x in ident if _fin(x)), default=None))
    tr = [s.get("trainer_signal_residual") for s in S]
    tn = [s.get("trainer_normalization_residual") for s in S if s.get("normalization_group") == "rollout"]
    add("reconstruction matches the trainer's own advantages",
        all(_fin(x) and 0 <= x <= TRAINER_TOL for x in tr + tn),
        {"signal": max((x for x in tr if _fin(x)), default=None), "normalized": max((x for x in tn if _fin(x)), default=None),
         "tolerance": TRAINER_TOL})
    gaps = [abs(s["rho"] - s["predicted_rho"]) if _fin(s.get("rho")) and _fin(s.get("predicted_rho")) else None
            for s in S]
    add("rho equals rho predicted before the update", all(g is not None and g <= 1e-5 for g in gaps),
        max((g for g in gaps if g is not None), default=None))
    rhos = [s.get("rho") for s in S]
    add("rho is a valid correlation", all(_fin(x) and -1.0 <= x <= 1.0 for x in rhos),
        [min((x for x in rhos if _fin(x)), default=None), max((x for x in rhos if _fin(x)), default=None)])
    add("eta labels correct", bool(eta0) and bool(eta1) and all(not r["eta_is_training_eta"] for r in eta0)
        and all(r["eta_is_training_eta"] and r["eta"] == eta for r in eta1),
        {"eta0_training": sorted({r["eta_is_training_eta"] for r in eta0}),
         "eta1_training": sorted({r["eta_is_training_eta"] for r in eta1})})
    with_terms = [r for r in recs if r.get("reward_terms")]
    sums = [sum(v for v in r["reward_term_abs_share"].values() if v is not None) for r in with_terms
            if any(v is not None for v in r["reward_term_abs_share"].values())]
    add("reward terms logged, shares sum to 1", bool(sums) and all(_fin(s) and abs(s - 1) < 1e-6 for s in sums),
        {"records_with_terms": len(with_terms), "terms": sorted(with_terms[-1]["reward_terms"]) if with_terms else []})
    to = sum(s.get("timeouts", 0) for s in S); te = sum(s.get("terminations", 0) for s in S)
    add("time limits and terminations both seen", to > 0 and te > 0, {"timeouts": to, "terminations": te})
    missing_groups = [r.get("iteration") for r in recs if r["signals"].get("groups") is None]
    add("statistics computed on the trainer's own normalization groups in every record", not missing_groups,
        {"records_without_groups": len(missing_groups),
         "normalization_group": sorted({s.get("normalization_group") for s in S}, key=str)})
    if control is not None:
        cr = [r["signals"].get("trainer_signal_residual") for r in control]
        worst = max((x for x in cr if _fin(x)), default=None)
        add("negative control caught (trainer that claims eta but ignores it)",
            len(control) == iterations and worst is not None and worst > CONTROL_MIN,
            {"records": len(control), "worst_residual": worst, "must_exceed": CONTROL_MIN})
    return out


def _run_dirs(root="logs"):
    return {d for d in glob.glob(os.path.join(root, "rsl_rl", "*", "*")) if os.path.isdir(d)}


def _last_checkpoint(run_dir):
    if not run_dir:
        return None
    ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
    def it(p):
        m = re.search(r"model_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=it) if ckpts else None


def _tensors(obj, prefix=""):
    import torch
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from _tensors(obj[k], f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _tensors(v, f"{prefix}/{i}")


def checkpoint_diff(a, b):
    """Compare every tensor in two checkpoints (weights and optimizer state)."""
    import torch
    ta = dict(_tensors(torch.load(a, map_location="cpu", weights_only=False)))
    tb = dict(_tensors(torch.load(b, map_location="cpu", weights_only=False)))
    common = sorted(set(ta) & set(tb))
    diffs = {}
    for k in common:
        x, y = ta[k], tb[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            diffs[k] = float("inf")
        elif not torch.equal(x, y):
            diffs[k] = float((x.double() - y.double()).abs().max())
    return {"tensors": len(common), "keys_match": set(ta) == set(tb), "differing": len(diffs),
            "max_abs_diff": max(diffs.values(), default=0.0), "first_differing": sorted(diffs)[:3]}


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def identity_check(stock_ckpt, repeat_ckpt, eta0_ckpt):
    """status: "passed" (stock reproducible and eta 0 identical), "failed" (stock reproducible, eta 0 differs) or
    "inconclusive" (two stock runs already differ, so a difference proves nothing and a match proves little)."""
    name = "eta 0 with Sim2Gate attached is bit-identical to stock (final checkpoints)"
    missing = [k for k, v in (("stock", stock_ckpt), ("stock_repeat", repeat_ckpt), ("eta0", eta0_ckpt)) if not v]
    if missing:
        return {"name": name, "ok": False, "status": "failed", "detail": f"checkpoint not found for {missing}"}
    rep_d = checkpoint_diff(stock_ckpt, repeat_ckpt)
    eta_d = checkpoint_diff(stock_ckpt, eta0_ckpt)
    reproducible = rep_d["keys_match"] and rep_d["differing"] == 0
    identical = eta_d["keys_match"] and eta_d["differing"] == 0
    detail = {"method": "every tensor in the checkpoint (model and optimizer state) compared with torch.equal; "
                        "file bytes are not compared",
              "stock_vs_repeat": rep_d, "stock_vs_eta0": eta_d,
              "checkpoints": {k: {"path": v, "sha256": _sha256(v)} for k, v in
                              (("stock", stock_ckpt), ("stock_repeat", repeat_ckpt), ("eta0", eta0_ckpt))}}
    if not reproducible:
        status = "inconclusive"
        detail["note"] = ("two stock runs with the same seed already differ, so this setup is not reproducible "
                          "run to run; identity cannot be shown by checkpoint comparison")
    else:
        status = "passed" if identical else "failed"
    return {"name": name, "ok": status == "passed", "status": status, "detail": detail}


def overhead_check(base_text, eta0_text, max_overhead=0.10):
    a, b = iteration_times(base_text), iteration_times(eta0_text)
    if not a or not b:
        return {"name": "monitor overhead", "ok": False, "detail": "iteration times not found in logs"}
    ratio = statistics.median(b) / statistics.median(a) - 1
    return {"name": f"monitor overhead under {int(max_overhead * 100)}%", "ok": ratio <= max_overhead,
            "detail": {"stock_s": round(statistics.median(a), 4), "with_sim2gate_s": round(statistics.median(b), 4),
                       "overhead": round(ratio, 4)}}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim2gate.training.smoke_go2", description=__doc__.splitlines()[0])
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--out", default=None, help="output folder (default sim2gate_smoke/<time>)")
    p.add_argument("--no-control", action="store_true", help="skip the negative-control run (saves a minute)")
    p.add_argument("--identity", action="store_true",
                   help="also repeat the stock run and require the eta 0 run to be bit-identical to stock")
    p.add_argument("--isaaclab-script", default=None,
                   help="Isaac Lab 3.0 beta only: path to scripts/reinforcement_learning/rsl_rl/train.py")
    args = p.parse_args(argv)
    out = args.out or os.path.join("sim2gate_smoke", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    py = isaac_python()
    common = ["--task", args.task, "--seed", str(args.seed), "--max_iterations", str(args.iterations)]
    if args.isaaclab_script:
        plain = py + [args.isaaclab_script, "--headless"] + common
        extra = ["--isaaclab-script", args.isaaclab_script]
        common = common + ["--headless"]
    else:
        plain = py + [train_script(), "--rl_library", "rsl_rl"] + common
        extra = []
    s0, s1 = os.path.join(out, "signals_eta0.jsonl"), os.path.join(out, "signals_eta1.jsonl")
    sc = os.path.join(out, "signals_control.jsonl")
    for f in (s0, s1, sc):
        if os.path.exists(f):
            os.remove(f)
    launch = py + ["-m", "sim2gate.training.isaaclab_launch"]
    rc, dirs = {}, {}

    def tracked(name, cmd, log):
        before = _run_dirs()
        code, text = run(cmd, log)
        new = sorted(_run_dirs() - before)
        dirs[name] = new[0] if len(new) == 1 else None
        return code, text

    rc["stock"], plain_text = tracked("stock", plain, os.path.join(out, "run_stock.log"))
    if args.identity:
        rc["stock_repeat"], _ = tracked("stock_repeat", plain, os.path.join(out, "run_stock_repeat.log"))
    rc["eta0"], eta0_text = tracked("eta0", launch + ["--eta", "0", "--signals", s0] + extra + ["--"] + common,
                                    os.path.join(out, "run_eta0.log"))
    rc["eta1"], _ = tracked("eta1", launch + ["--eta", str(args.eta), "--signals", s1] + extra + ["--"] + common,
                            os.path.join(out, "run_eta1.log"))
    control = None
    if not args.no_control:
        rc["control"], _ = tracked("control", launch + ["--eta", str(args.eta), "--negative-control", "--signals", sc]
                                   + extra + ["--"] + common, os.path.join(out, "run_control.log"))
        control = load(sc)
    eta0, eta1 = load(s0), load(s1)
    results = checks(args.iterations, eta0, eta1, rc, eta=args.eta, control=control)
    results.append(overhead_check(plain_text, eta0_text))
    if args.identity:
        results.append(identity_check(_last_checkpoint(dirs.get("stock")), _last_checkpoint(dirs.get("stock_repeat")),
                                      _last_checkpoint(dirs.get("eta0"))))
    from .summary import summarize
    summaries = [summarize(f, 0.2) for f in (s0, s1) if load(f)]
    try:
        import importlib.metadata as m
        versions = {k: m.version(k) for k in ("sim2gate", "rsl-rl-lib")}
    except Exception:
        versions = {}
    failed = [r for r in results if not r["ok"] and r.get("status") != "inconclusive"]
    inconclusive = [r for r in results if r.get("status") == "inconclusive"]
    result = "FAIL" if failed else "INCONCLUSIVE" if inconclusive else "PASS"
    report = {"result": result, "task": args.task, "iterations": args.iterations,
              "run_dirs": dirs,
              "seed": args.seed, "eta": args.eta, "versions": versions, "checks": results, "summaries": summaries}
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1)
    width = shutil.get_terminal_size((100, 20)).columns
    print("\n" + "=" * min(width, 100))
    for r in results:
        label = "INCONCLUSIVE" if r.get("status") == "inconclusive" else "PASS" if r["ok"] else "FAIL"
        print(f"{label}  {r['name']}: {r['detail']}")
    for s in summaries:
        omr = s["mean_one_minus_rho"]
        print(f"signals  eta={s['eta']} training={s['eta_is_training_eta']}: "
              f"1-rho={'n/a' if omr is None else f'{omr:.4f}'} reversal={s['reversal_share'] or 0:.2f} "
              f"zero_tail={s['mean_zero_tail_fraction']:.2f}  (last 20% of iterations)")
    if dirs.get("eta1"):
        print(f"eta={args.eta} policy for check 2: --checkpoint {_last_checkpoint(dirs['eta1']) or dirs['eta1']}")
    print(f"\nSMOKE TEST {report['result']}  (report: {os.path.join(out, 'report.json')})")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[result]


if __name__ == "__main__":
    sys.exit(main())
