"""Run Isaac Lab's own RSL-RL training with Sim2Gate's TrainingMonitor attached and, optionally, BSRS shaping.

Nothing in Isaac Lab is edited. This launcher patches rsl_rl's OnPolicyRunner so that, when Isaac Lab builds the
runner, (1) the algorithm class is swapped for BSRSPPO with the requested eta (only if --eta is not 0), and
(2) a TrainingMonitor is attached, writing one JSON line per PPO iteration. Then it hands the remaining arguments
to Isaac Lab's training entry point unchanged.

Requires Isaac Lab 3.x with rsl-rl-lib 5.x. The math was gated on rsl-rl-lib 5.5.1 (Isaac Lab 3.0.0's pin),
5.4.1 (the Isaac Lab inside Isaac Lab-Arena) and 5.0.1 (Isaac Lab 3.0 beta's). Single GPU only (every rank would write its own file).

Isaac Lab 3.0.0 (unified entry point; this is the default):
    ./isaaclab.sh -p -m sim2gate.training.isaaclab_launch --eta 1.0 -- \\
        --task Isaac-Velocity-Flat-UnitreeGo2 --seed 1 --max_iterations 20

Isaac Lab 3.0 beta (per-library script, task ids ending in -v0): add --isaaclab-script:
    ./isaaclab.sh -p -m sim2gate.training.isaaclab_launch --eta 1.0 \\
        --isaaclab-script scripts/reinforcement_learning/rsl_rl/train.py -- \\
        --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless --seed 1 --max_iterations 20
"""
import argparse
import importlib.metadata as metadata
import os
import runpy
import sys

GATED_RSL_RL = {"5.0.1", "5.4.1", "5.5.1"}


def _split(argv):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def install(eta=0.0, signals=None, what_if_eta=1.0, monitor=True, negative_control=False, evidence=None):
    """Patch rsl_rl.runners.OnPolicyRunner. Returns a function that undoes the patch.

    evidence: None, or keyword arguments for EvidenceRecorder (out_dir required) to attach the v2 phase 0A recorder
    after the monitor."""
    version = metadata.version("rsl-rl-lib")
    if int(version.split(".")[0]) < 5:
        raise RuntimeError(f"rsl-rl-lib {version} found; Sim2Gate's training hooks need rsl-rl-lib 5.x (Isaac Lab 3.x)")
    if version not in GATED_RSL_RL:
        print(f"[sim2gate] warning: rsl-rl-lib {version} has not been through the acceptance gate "
              f"(gated: {', '.join(sorted(GATED_RSL_RL))})", file=sys.stderr)
    from rsl_rl.runners import OnPolicyRunner
    from .monitor import TrainingMonitor

    orig = OnPolicyRunner.__init__

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if eta != 0.0:
            train_cfg["algorithm"]["class_name"] = ("sim2gate.training.controls:IgnoresEtaBSRSPPO" if negative_control
                                                    else "sim2gate.training.bsrs_ppo:BSRSPPO")
            train_cfg["algorithm"]["eta"] = float(eta)
        orig(self, env, train_cfg, log_dir=log_dir, device=device)
        if monitor:
            path = signals or os.path.join(log_dir or ".", "sim2gate_signals.jsonl")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.sim2gate_monitor = TrainingMonitor(self, path, eta=what_if_eta)
            print(f"[sim2gate] training eta={eta}; signals -> {path}", file=sys.stderr)
        if evidence:
            from .evidence import EvidenceRecorder
            self.sim2gate_evidence = EvidenceRecorder(self, **evidence)
            print(f"[sim2gate] evidence -> {evidence['out_dir']}", file=sys.stderr)

    OnPolicyRunner.__init__ = __init__
    return lambda: setattr(OnPolicyRunner, "__init__", orig)


def _ints(text):
    return tuple(int(x) for x in text.split(",") if x.strip())


def _evidence_kwargs(args, rest):
    if not args.evidence:
        return None
    kw = {"out_dir": args.evidence, "sample_envs": args.evidence_sample_envs,
          "refit_terms": tuple(t.strip() for t in args.evidence_refit_terms.split(",") if t.strip()),
          "refit_iterations": _ints(args.evidence_refit_iterations),
          "meta": {"launcher": {"eta": args.eta, "isaaclab_args": rest}}}
    if args.evidence_deep:
        kw["deep"] = _ints(args.evidence_deep)
    if args.evidence_grad_save:
        kw["grad_save"] = _ints(args.evidence_grad_save)
    if bool(kw["refit_terms"]) != bool(kw["refit_iterations"]):
        raise SystemExit("--evidence-refit-terms and --evidence-refit-iterations go together")
    return kw


def main(argv=None):
    ours, rest = _split(list(sys.argv[1:] if argv is None else argv))
    p = argparse.ArgumentParser(prog="python -m sim2gate.training.isaaclab_launch",
                                description="Isaac Lab RSL-RL training with Sim2Gate signals. "
                                            "Arguments after -- go to Isaac Lab unchanged.")
    p.add_argument("--eta", type=float, default=0.0, help="BSRS shaping strength; 0 = stock PPO (default)")
    p.add_argument("--signals", default=None, help="JSONL output (default: <run log dir>/sim2gate_signals.jsonl)")
    p.add_argument("--what-if-eta", type=float, default=1.0,
                   help="eta used for the what-if signals when training with stock PPO (default 1.0)")
    p.add_argument("--isaaclab-script", default=None,
                   help="path to Isaac Lab's rsl_rl/train.py (3.0 beta); omit to use isaaclab_rl's entry point")
    p.add_argument("--no-monitor", action="store_true", help="only swap in BSRSPPO, log nothing extra")
    p.add_argument("--negative-control", action="store_true",
                   help="acceptance testing only: a trainer that claims --eta but ignores it")
    p.add_argument("--update-identity", default=None, metavar="OUT.json",
                   help="acceptance testing only: replay the first PPO update stock, stock again and with Sim2Gate "
                        "attached, from the same state, and write the comparison (stock PPO, eta 0, no monitor "
                        "otherwise; use --max_iterations 1)")
    p.add_argument("--evidence", default=None, metavar="DIR",
                   help="v2 phase 0A: write the per-reward-term evidence record to DIR (see docs/v2/PHASE0A.md)")
    p.add_argument("--evidence-deep", default=None, metavar="ITERS",
                   help="comma-separated iterations for gradient directions and sample dumps (default: a fixed list)")
    p.add_argument("--evidence-grad-save", default=None, metavar="ITERS",
                   help="comma-separated deep iterations whose full gradient vectors are saved (default: a fixed list)")
    p.add_argument("--evidence-sample-envs", type=int, default=256, help="environments in each sample dump")
    p.add_argument("--evidence-refit-terms", default="", metavar="TERMS",
                   help="comma-separated reward terms to refit the critic without (default: none)")
    p.add_argument("--evidence-refit-iterations", default="", metavar="ITERS",
                   help="comma-separated iterations at which to run the critic refits")
    args = p.parse_args(ours)
    if not args.isaaclab_script:
        # as Isaac Lab 3.0's train.py does: set before anything defines Warp kernels (halves cold kernel builds)
        try:
            import warp as wp
            wp.config.enable_backward = False
        except ImportError:
            pass
    if args.negative_control and args.eta == 0.0:
        p.error("--negative-control needs a nonzero --eta")
    if args.update_identity:
        if args.eta != 0.0 or args.negative_control:
            p.error("--update-identity compares stock PPO with Sim2Gate attached at eta 0; leave --eta at 0")
        from .update_identity import install as install_identity
        install_identity(args.update_identity)
    else:
        install(args.eta, args.signals, args.what_if_eta, monitor=not args.no_monitor,
                negative_control=args.negative_control, evidence=_evidence_kwargs(args, rest))
    if args.isaaclab_script:
        script = os.path.abspath(args.isaaclab_script)
        sys.path.insert(0, os.path.dirname(script))       # train.py imports its sibling cli_args.py
        sys.argv = [script] + rest
        runpy.run_path(script, run_name="__main__")
    else:
        from isaaclab_rl.entrypoints.backends.train_rsl_rl import run
        run(rest)


if __name__ == "__main__":
    main()
