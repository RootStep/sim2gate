"""v2 phase 0A: the evidence pack for Vihaan's decision on extending rho to ordinary PPO reward terms.

Ten Go2 flat-ground training runs on Isaac Lab + RSL-RL with sim2gate.training.evidence.EvidenceRecorder attached
(nothing about training changes; tests/test_evidence.py checks bit-identical parameters), then the Layer 1 outcome of
each finished policy, then one bundle. Arms, 2 seeds each (seeds 1 and 2), all on the P0 v4 recipe (2000 iterations):

    base            P0 v4 as is (also BSRS at eta = 0)
    no_slip         P0 v4 without the foot-slip penalty (rewards.feet_slide); v1: foot slip about 25 times higher
    no_action_rate  P0 v4 without the action-rate penalty (rewards.action_rate_l2), as v1's P4; jitter about 25% higher
    bsrs_0.5        P0 v4 trained with same-critic BSRS, eta = 0.5 (the case paper V1.4 covers)
    bsrs_2          the same, eta = 2

Critic refits (the critic issue, v2 item 6) on base seed 1, without feet_slide and without action_rate_l2.
Nothing here is blinded: phase 0A is input to a mathematical decision, not a test of Sim2Gate. The data format is in
docs/v2/PHASE0A.md.

Inside the Isaac Lab container, from the Isaac Lab-Arena folder:
    python -m sim2gate.v2.phase0a smoke    --work v2_runs/phase0a_smoke   # ~10 min: one short run, checks the record
    python -m sim2gate.v2.phase0a train    --work v2_runs/phase0a         # the 10 runs (resumable)
    python -m sim2gate.v2.phase0a outcomes --work v2_runs/phase0a         # check 2 and walking check per run
    python -m sim2gate.v2.phase0a bundle   --work v2_runs/phase0a --out phase0a-evidence.tgz
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
import tarfile

ITERATIONS = 2000                                  # P0 v4
SEEDS = (1, 2)
ARMS = {
    "base": {"task": "P0", "eta": 0.0},
    "no_slip": {"task": "Sim2Gate-V2-0A-Go2-NoSlip-v1", "eta": 0.0, "removed_term": "feet_slide"},
    "no_action_rate": {"task": "Sim2Gate-V2-0A-Go2-NoActionRate-v1", "eta": 0.0, "removed_term": "action_rate_l2"},
    "bsrs_0.5": {"task": "P0", "eta": 0.5},
    "bsrs_2": {"task": "P0", "eta": 2.0},
}
REFIT = {"run": ("base", 1), "terms": ("feet_slide", "action_rate_l2"), "iterations": (50, 200, 500, 1000, 1999)}
DEEP = (0, 1, 2, 5, 10, 20, 50) + tuple(range(100, ITERATIONS, 100)) + (ITERATIONS - 1,)
GRAD_SAVE = (0, 50, 200, 500, 1000, 1999)
SAMPLE_ENVS = 256
CALLBACK = ["--external_callback", "sim2gate.v2.phase0a.register"]
_VARIANTS = {
    "Sim2Gate-V2-0A-Go2-NoSlip-v1": {"remove": "feet_slide", "env_class": "Go2FlatV2NoSlipEnvCfg",
                                     "agent_class": "Go2FlatV2NoSlipPPORunnerCfg",
                                     "experiment_name": "sim2gate_v2_0a_go2_flat_no_slip"},
    "Sim2Gate-V2-0A-Go2-NoActionRate-v1": {"remove": "action_rate_l2", "env_class": "Go2FlatV2NoActionRateEnvCfg",
                                           "agent_class": "Go2FlatV2NoActionRatePPORunnerCfg",
                                           "experiment_name": "sim2gate_v2_0a_go2_flat_no_action_rate"},
}


def task_of(arm):
    from ..mvp import tasks
    t = ARMS[arm]["task"]
    return tasks.P0_TASK_V4 if t == "P0" else t


# ---- task variants (Isaac Lab entry points, built on first use) -------------------------------------------------

def _env_cfg(remove):
    from isaaclab.utils import configclass
    from ..mvp import tasks
    base = tasks._p0_env_cfg(4)

    @configclass
    class Go2FlatV2RemovedTermEnvCfg(base):
        def __post_init__(self):
            super().__post_init__()
            if getattr(self.rewards, remove, None) is None:
                raise ValueError(f"P0 v4 has no reward term {remove}")
            setattr(self.rewards, remove, None)

    return Go2FlatV2RemovedTermEnvCfg


def _agent_cfg(experiment_name):
    from isaaclab.utils import configclass
    from ..mvp import tasks
    base = tasks._p0_agent_cfg(tasks.P0_TASK_V4)

    @configclass
    class Go2FlatV2PPORunnerCfg(base):
        def __post_init__(self):
            super().__post_init__()
            self.experiment_name = experiment_name
            self.max_iterations = ITERATIONS

    return Go2FlatV2PPORunnerCfg


_built = {}


def __getattr__(name):
    for v in _VARIANTS.values():
        if name in (v["env_class"], v["agent_class"]):
            if name not in _built:
                _built[v["env_class"]] = _env_cfg(v["remove"])
                _built[v["agent_class"]] = _agent_cfg(v["experiment_name"])
            return _built[name]
    raise AttributeError(name)


def register():
    """Isaac Lab --external_callback: the MVP tasks (P0 v4 included) and the two phase 0A variants."""
    import gymnasium as gym
    from ..mvp import tasks
    tasks.register()
    for task, v in _VARIANTS.items():
        if task not in gym.registry:
            gym.register(id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                         kwargs={"env_cfg_entry_point": f"{__name__}:{v['env_class']}",
                                 "rsl_rl_cfg_entry_point": f"{__name__}:{v['agent_class']}",
                                 "default_agent": "rsl_rl", "sim2gate_variant": {"phase": "v2-0A", **v}})
    return None


# ---- commands ------------------------------------------------------------------------------------------------------

def runs():
    return [(arm, seed) for arm in ARMS for seed in SEEDS]


def run_name(arm, seed):
    return f"{arm}_s{seed}"


def _csv(xs):
    return ",".join(str(x) for x in xs)


def train_command(arm, seed, work, iterations=ITERATIONS, deep=DEEP, grad_save=GRAD_SAVE, refit=None):
    from ..checks import transfer
    from ..training.smoke_go2 import isaac_python
    d = os.path.join(work, "runs", run_name(arm, seed))
    refit = refit if refit is not None else (REFIT if REFIT["run"] == (arm, seed) else None)
    cmd = isaac_python() + ["-m", "sim2gate.training.isaaclab_launch", "--eta", str(ARMS[arm]["eta"]),
                            "--signals", os.path.join(d, "monitor.jsonl"), "--evidence", os.path.join(d, "evidence"),
                            "--evidence-deep", _csv(x for x in deep if x < iterations),
                            "--evidence-grad-save", _csv(x for x in grad_save if x < iterations),
                            "--evidence-sample-envs", str(SAMPLE_ENVS)]
    if refit:
        cmd += ["--evidence-refit-terms", _csv(refit["terms"]),
                "--evidence-refit-iterations", _csv(x for x in refit["iterations"] if x < iterations)]
    return cmd + ["--", "--task", task_of(arm), "--seed", str(seed), "--max_iterations", str(iterations)] \
        + CALLBACK + [transfer.PHYSX]


def outcome_commands(arm, ckpt, out_dir):
    from ..checks import transfer
    from ..mvp.baselines import GO2_COMMANDS
    from ..mvp.blinded import CHECK2
    from ..training.smoke_go2 import isaac_python
    task = ["--task", task_of(arm)]
    o = {"check2": os.path.join(out_dir, "check2.json"), "nominal": os.path.join(out_dir, "nominal.json")}
    return {
        "check2": isaac_python() + ["-m", "sim2gate.checks.isaaclab_exploits", "--steps", str(CHECK2["steps"]),
                                    "--commands", GO2_COMMANDS, "--out", o["check2"], "--"] + task
        + ["--num_envs", str(CHECK2["num_envs"]), "--checkpoint", ckpt] + CALLBACK + [CHECK2["physics"]],
        "nominal": isaac_python() + ["-m", "sim2gate.checks.isaaclab_conditions", "--mode", "nominal", "--commands",
                                     GO2_COMMANDS, "--out", o["nominal"], "--"] + task + ["--checkpoint", ckpt]
        + CALLBACK + [transfer.PHYSX] + list(transfer.FIXED_OVERRIDES) + list(transfer.ORDERING),
    }, o


# ---- state ---------------------------------------------------------------------------------------------------------

def _say(msg):
    print(f"[sim2gate] {msg}", flush=True)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _state_path(work):
    return os.path.join(work, "state.json")


def load_state(work):
    p = _state_path(work)
    return json.load(open(p)) if os.path.exists(p) else {"created": _now(), "runs": {}}


def save_state(work, state):
    tmp = _state_path(work) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, _state_path(work))


def check_record(evidence_dir, iterations, terms_expected=None):
    """Problems with one run's evidence record (empty list = fine)."""
    problems = []
    meta_p, upd_p = os.path.join(evidence_dir, "meta.json"), os.path.join(evidence_dir, "updates.jsonl")
    if not (os.path.exists(meta_p) and os.path.exists(upd_p)):
        return ["no evidence record"]
    meta = json.load(open(meta_p))
    recs = [json.loads(x) for x in open(upd_p) if x.strip()]
    if len(recs) != iterations:
        problems.append(f"{len(recs)} update records, expected {iterations}")
    if terms_expected is not None and meta["reward_terms"] != terms_expected:
        problems.append(f"reward terms {meta['reward_terms']}")
    worst = lambda k: max((r[k] for r in recs if r.get(k) is not None), default=None)  # noqa: E731
    ts, ir, tr = worst("term_sum_residual"), worst("identity_residual"), worst("trainer_residual")
    # terms vs reward: float32 sums of ~12 terms; A = sum T_k + V: float32 recursions over 24 steps
    if ts is None or ts > 1e-4:
        problems.append(f"reward terms do not add up to the reward (max residual {ts})")
    if ir is None or ir > 1e-2:
        problems.append(f"A = sum_k T_k + V does not hold (max residual {ir})")
    if tr is None or tr > 1e-4:
        problems.append(f"recorded advantages differ from the trainer's (max relative residual {tr})")
    deep = [r for r in recs if r.get("deep")]
    want_deep = [i for i in meta["deep_iterations"] if i < iterations]
    if [r["iteration"] for r in deep] != want_deep:
        problems.append("deep iterations missing")
    for i in want_deep:
        if not os.path.exists(os.path.join(evidence_dir, "deep", f"it{i:05d}_samples.npz")):
            problems.append(f"sample dump {i} missing")
    for i in meta["grad_save_iterations"]:
        if i < iterations and not os.path.exists(os.path.join(evidence_dir, "deep", f"it{i:05d}_directions.npz")):
            problems.append(f"directions {i} missing")
    return problems


# ---- steps ---------------------------------------------------------------------------------------------------------

def _train(arm, seed, work, runner, iterations=ITERATIONS, **kw):
    from ..mvp.baselines import shutdown_crash
    from ..training.smoke_go2 import _last_checkpoint, _run_dirs
    name = run_name(arm, seed)
    d = os.path.join(work, "runs", name)
    if os.path.exists(d):                       # a half-finished attempt: keep it for the record, start clean
        os.replace(d, d + f".incomplete-{datetime.datetime.now():%Y%m%d%H%M%S}")
    os.makedirs(d)
    before = _run_dirs()
    t0 = _now()
    rc, text = runner(train_command(arm, seed, work, iterations=iterations, **kw), os.path.join(d, "train.log"))
    new = sorted(_run_dirs() - before)
    ck = _last_checkpoint(new[0]) if len(new) == 1 else None
    rec = {"arm": arm, "seed": seed, "task": task_of(arm), "eta": ARMS[arm]["eta"], "iterations": iterations,
           "started": t0, "finished": _now(), "train_returncode": rc,
           "run_dir": os.path.abspath(new[0]) if new else None}
    crash = shutdown_crash(rc, text, ck, iterations) if ck else None
    if crash:
        rec["train_shutdown_crash"] = crash
    problems = []
    err = os.path.join(d, "evidence", "error.txt")
    if os.path.exists(err):                     # the recorder's own traceback (Isaac Sim may not print it)
        lines = [x for x in open(err).read().splitlines() if x.strip()]
        problems.append("recorder error: " + " | ".join(lines[:1] + lines[-1:]))
    if ck is None or (rc != 0 and not crash):
        problems.append(f"training exited {rc}; {len(new)} new run folder(s)")
    else:
        rec["checkpoint"] = {"path": os.path.abspath(ck), "sha256": _sha256(ck)}
        problems += check_record(os.path.join(d, "evidence"), iterations)
    rec["problems"] = problems
    rec["ok"] = not problems
    return rec


def train_all(work, runner, only=None):
    os.makedirs(work, exist_ok=True)
    state = load_state(work)
    state.setdefault("plan", {"arms": ARMS, "seeds": list(SEEDS), "iterations": ITERATIONS, "refit": REFIT})
    todo = [r for r in runs() if (only is None or run_name(*r) in only)
            and not state["runs"].get(run_name(*r), {}).get("ok")]
    for i, (arm, seed) in enumerate(todo, 1):
        _say(f"phase 0A run {i}/{len(todo)}: {run_name(arm, seed)} (output in {work}/runs)")
        rec = _train(arm, seed, work, runner)
        state["runs"][run_name(arm, seed)] = rec
        save_state(work, state)
        _say(f"  {'ok' if rec['ok'] else 'PROBLEM: ' + '; '.join(rec['problems'])}")
    bad = [n for n, r in state["runs"].items() if not r.get("ok")]
    _say("ALL RUNS OK" if not bad and len(state["runs"]) == len(runs()) else f"not ok: {bad}")
    return 0 if not bad else 1


def outcomes(work, runner):
    from ..checks import robustness
    from ..mvp.blinded import FROZEN_SUCCESS
    state = load_state(work)
    success = json.load(open(FROZEN_SUCCESS))
    for name, rec in sorted(state["runs"].items()):
        if not rec.get("ok"):
            continue
        out = os.path.join(work, "runs", name, "outcomes")
        os.makedirs(out, exist_ok=True)
        cmds, files = outcome_commands(rec["arm"], rec["checkpoint"]["path"], out)
        for m, cmd in cmds.items():
            if not os.path.exists(files[m]):
                _say(f"{name}: {m}")
                runner(cmd, os.path.join(out, f"{m}.log"))
        o = {}
        if os.path.exists(files["check2"]):
            mt = json.load(open(files["check2"]))["metrics"]
            o.update(foot_slip_p95_mps=mt.get("foot_slip_p95_mps"),
                     action_jitter_high_freq_power_share=mt.get("action_jitter_high_freq_power_share"),
                     contact_force_p99_bw=mt.get("contact_force_p99_bw"))
        if os.path.exists(files["nominal"]):
            raw = json.load(open(files["nominal"]))
            o["nominal_success_rate"] = robustness.success_rate(raw["episodes"], success["threshold_mps"])
            o["nominal_falls"] = sum(bool(e["fell"]) for e in raw["episodes"])
            o["nominal_episodes"] = len(raw["episodes"])
            o["checkpoint_matches"] = raw.get("checkpoint_sha256") == rec["checkpoint"]["sha256"]
        rec["outcomes"] = o
        save_state(work, state)
        _say(f"{name}: {o}")
    return 0


def bundle(work, out):
    """One .tgz with every run's evidence, monitor log, training log, outcomes and final checkpoint, plus
    MANIFEST.json (SHA-256 of every file) and the data-format spec."""
    state = load_state(work)
    files = []
    for name in sorted(state["runs"]):
        d = os.path.join(work, "runs", name)
        for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if os.path.isfile(p):
                files.append((p, os.path.relpath(p, work)))
        ck = (state["runs"][name].get("checkpoint") or {}).get("path")
        if ck and os.path.exists(ck):
            files.append((ck, os.path.join("runs", name, "checkpoint", os.path.basename(ck))))
    spec = os.path.join(os.path.dirname(__file__), "PHASE0A_FORMAT.md")
    if os.path.exists(spec):
        files.append((spec, "FORMAT.md"))
    manifest = {"created": _now(), "state": state, "files": {arc: _sha256(p) for p, arc in files}}
    mp = os.path.join(work, "MANIFEST.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=1)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(mp, arcname="phase0a/MANIFEST.json")
        for p, arc in files:
            tar.add(p, arcname=os.path.join("phase0a", arc))
    _say(f"bundle {out}: {len(files)} files, {os.path.getsize(out) / 1e6:.0f} MB, sha256 {_sha256(out)}")
    return 0


def smoke(work, runner, iterations=12):
    """A short base run with every recorder feature on (deep 0-2 and 10-11, refits at 1), checked like a real run."""
    os.makedirs(work, exist_ok=True)
    rec = _train("base", 99, work, runner, iterations=iterations, deep=(0, 1, 2, 10, 11), grad_save=(0, 11),
                 refit={"terms": REFIT["terms"], "iterations": (1,)})
    d = os.path.join(work, "runs", run_name("base", 99), "evidence")
    if os.path.exists(os.path.join(d, "updates.jsonl")):
        recs = [json.loads(x) for x in open(os.path.join(d, "updates.jsonl"))]
        meta = json.load(open(os.path.join(d, "meta.json")))
        _say(f"reward terms: {meta['reward_terms']}")
        last = recs[-1]
        _say(f"term shares (last): {dict(zip(meta['reward_terms'], [round(x, 3) for x in last['term_abs_share']]))}")
        _say(f"residuals (max): term sum {max(r['term_sum_residual'] for r in recs):.2e}, identity "
             f"{max(r['identity_residual'] for r in recs):.2e}, trainer {max(r['trainer_residual'] for r in recs):.2e}")
        if recs[1].get("refit"):
            _say(f"refit at 1: {json.dumps(recs[1]['refit'])}")
        _say(f"cos with trainer direction (it 11): {json.dumps(recs[-1].get('direction_cos_trainer'))}")
        size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                   if os.path.isfile(p))
        _say(f"record size {size / 1e6:.1f} MB")
    _say("SMOKE OK" if rec["ok"] else "SMOKE FAILED: " + "; ".join(rec["problems"]))
    json.dump(rec, open(os.path.join(work, "smoke.json"), "w"), indent=1)
    return 0 if rec["ok"] else 1


def main(argv=None):
    from ..mvp.testset import quiet_run
    p = argparse.ArgumentParser(prog="python -m sim2gate.v2.phase0a", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("smoke", "train", "outcomes", "bundle"):
        s = sub.add_parser(c)
        s.add_argument("--work", required=True)
        if c == "train":
            s.add_argument("--only", nargs="*", default=None, help="run names, e.g. base_s1 no_slip_s2")
        if c == "bundle":
            s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "smoke":
        return smoke(a.work, quiet_run)
    if a.cmd == "train":
        return train_all(a.work, quiet_run, a.only)
    if a.cmd == "outcomes":
        return outcomes(a.work, quiet_run)
    return bundle(a.work, a.out)


if __name__ == "__main__":
    sys.exit(main())
