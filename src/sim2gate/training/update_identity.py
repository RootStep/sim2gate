"""Update-level identity: does attaching Sim2Gate change what one PPO update produces, given the same input?

Isaac Lab on GPU is not bit-reproducible run to run (two stock runs with the same seed differ), so final
checkpoints cannot show that Sim2Gate at eta 0 leaves training unchanged. This check removes the simulator from
the comparison. It collects one real rollout, then, from that exact state (every network, optimizer and storage
tensor, the algorithm's scalar settings, and the CPU and CUDA random-number states), runs RSL-RL's
compute_returns + update three times:

    A   stock
    A2  stock again        A == A2 shows the replay itself is deterministic on this GPU
    B   with Sim2Gate's TrainingMonitor attached (its compute_returns, minibatch-generator and update wrappers)

and compares every tensor afterwards (actor, critic, optimizer state, storage returns and advantages) plus the loss
values and learning rate. Result: passed (A == A2 and A == B), failed (A == A2, A != B), or inconclusive
(A != A2). Training then continues normally from the restored state, so the run ends like any other.

Not covered here: the monitor's per-step wrapper (process_env_step) runs during the rollout, which happens once.
It is a pass-through call; the CPU test of full training (tests/test_monitor.py) covers it bit for bit.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m sim2gate.training.isaaclab_launch \\
        --update-identity sim2gate_update_identity.json -- \\
        --task Isaac-Velocity-Flat-UnitreeGo2 --seed 1 --max_iterations 1
"""
import copy
import json
import os
import sys
import tempfile

import torch


def _modules(alg):
    return {k: v for k, v in vars(alg).items() if isinstance(v, torch.nn.Module)}


def _optimizers(alg):
    return {k: v for k, v in vars(alg).items() if isinstance(v, torch.optim.Optimizer)}


def _scalars(obj):
    return {k: v for k, v in vars(obj).items() if isinstance(v, (bool, int, float, str, type(None)))}


def _storage_tensors(st):
    return {k: v for k, v in vars(st).items() if torch.is_tensor(v)}


def snapshot(alg):
    st = alg.storage
    return {
        "modules": {k: {n: t.detach().clone() for n, t in list(m.named_parameters()) + list(m.named_buffers())}
                    for k, m in _modules(alg).items()},
        "optimizers": {k: copy.deepcopy(o.state_dict()) for k, o in _optimizers(alg).items()},
        "alg_scalars": _scalars(alg),
        "storage": {k: v.detach().clone() for k, v in _storage_tensors(st).items()},
        "storage_scalars": _scalars(st),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _copy_into(t, v):
    """In place, so every tensor object (and anything aliasing it) stays the same. Tensors written during the
    rollout are inference tensors, which can only be written in inference mode."""
    with torch.no_grad():
        if t.is_inference():
            with torch.inference_mode():
                t.copy_(v)
        else:
            t.copy_(v)


def _load_module(m, sd):
    own = dict(m.named_parameters())
    own.update(dict(m.named_buffers()))
    if set(own) != set(sd):
        raise RuntimeError(f"update identity: {type(m).__name__} state changed shape: "
                           f"{sorted(set(own) ^ set(sd))[:5]}")
    for k, t in own.items():
        _copy_into(t, sd[k])


def restore(alg, s):
    for k, m in _modules(alg).items():
        _load_module(m, s["modules"][k])
    for k, o in _optimizers(alg).items():
        o.load_state_dict(copy.deepcopy(s["optimizers"][k]))
    for k, v in s["alg_scalars"].items():
        setattr(alg, k, v)
    st = alg.storage
    for k, v in s["storage"].items():
        _copy_into(getattr(st, k), v)
    for k, v in s["storage_scalars"].items():
        setattr(st, k, v)
    torch.set_rng_state(s["rng_cpu"])
    if s["rng_cuda"] is not None:
        torch.cuda.set_rng_state_all(s["rng_cuda"])


def _flat(prefix, obj, out):
    if torch.is_tensor(obj):
        out[prefix] = obj.detach().clone()
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _flat(f"{prefix}/{k}", obj[k], out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flat(f"{prefix}/{i}", v, out)


def outcome(alg, loss):
    """Everything the update produced: tensors to compare, and scalars."""
    t = {}
    for k, m in _modules(alg).items():
        _flat(f"{k}", m.state_dict(), t)
    for k, o in _optimizers(alg).items():
        _flat(f"{k}", o.state_dict()["state"], t)
    st = alg.storage
    for k in ("returns", "advantages", "values"):
        if torch.is_tensor(getattr(st, k, None)):
            t[f"storage/{k}"] = getattr(st, k).detach().clone()
    scal = {"learning_rate": float(alg.learning_rate)}
    scal.update({f"loss/{k}": float(v) for k, v in (loss or {}).items() if v is not None})
    return t, scal


def compare(a, b):
    (ta, sa), (tb, sb) = a, b
    common = sorted(set(ta) & set(tb))
    diffs = {}
    for k in common:
        x, y = ta[k], tb[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            diffs[k] = "shape/dtype"
        elif not torch.equal(x, y):
            diffs[k] = float((x.double() - y.double()).abs().max())
    sdiff = {k: [sa.get(k), sb.get(k)] for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)}
    return {"tensors": len(common), "keys_match": set(ta) == set(tb), "differing": len(diffs),
            "first_differing": sorted(diffs)[:5],
            "max_abs_diff": max((v for v in diffs.values() if isinstance(v, float)), default=0.0),
            "scalars_compared": len(set(sa) | set(sb)), "scalars_differing": sdiff}


def _same(c):
    return c["keys_match"] and c["differing"] == 0 and not c["scalars_differing"]


def _iteration(alg, obs):
    """What OnPolicyRunner.learn does after the rollout: compute_returns in inference mode, then update."""
    with torch.inference_mode():
        alg.compute_returns(obs)
    return alg.update()


def run_replays(runner, obs, s0, monitor_cls, monitor_path):
    """s0: snapshot taken with the rollout complete and compute_returns not yet run; obs: the last observation.
    Must be called outside inference mode. Leaves the algorithm restored to s0. Returns the result document."""
    alg = runner.alg
    det_was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        def stock():
            restore(alg, s0)
            return outcome(alg, _iteration(alg, obs))

        a = stock()
        a2 = stock()
        restore(alg, s0)
        mon = monitor_cls(runner, monitor_path, writer=False)
        try:
            b = outcome(alg, _iteration(alg, obs))
        finally:
            mon.close()
        mon_records = sum(1 for _ in open(monitor_path))
    finally:
        torch.use_deterministic_algorithms(det_was)
        restore(alg, s0)
    rep, ident = compare(a, a2), compare(a, b)
    status = ("inconclusive" if not _same(rep) else "passed" if _same(ident) else "failed")
    doc = {"status": status,
           "name": "one PPO update from the same state: stock vs Sim2Gate attached at eta 0",
           "method": "snapshot after one real rollout (networks, optimizer, storage, algorithm scalars, CPU and CUDA "
                     "RNG); compute_returns + update replayed: stock, stock again, with TrainingMonitor; every "
                     "tensor compared with torch.equal; deterministic algorithms on (warn_only)",
           "stock_vs_stock": rep, "stock_vs_sim2gate": ident,
           "monitor_records_written": mon_records,
           "normalize_advantage_per_mini_batch": bool(getattr(alg, "normalize_advantage_per_mini_batch", False)),
           "num_transitions": int(alg.storage.num_transitions_per_env * alg.storage.num_envs),
           "device": str(getattr(alg, "device", "")),
           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           "not_covered": "the monitor's process_env_step wrapper (rollout-time pass-through; covered by the CPU "
                          "full-training identity test)"}
    if status == "inconclusive":
        doc["note"] = "replaying the same update twice already differs on this device, so identity cannot be shown"
    if mon_records != 1:
        doc["status"] = "failed"
        doc["note"] = f"the monitor wrote {mon_records} records for one update; expected 1 (was it attached?)"
    return doc


def install(out_path):
    """Patch OnPolicyRunner: the first compute_returns (inside the runner's inference mode) snapshots the state,
    the first update (outside it) runs the replays, writes out_path, restores the state and continues with a
    normal stock iteration. Returns an undo function."""
    from rsl_rl.runners import OnPolicyRunner
    from .monitor import TrainingMonitor

    orig_init = OnPolicyRunner.__init__

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        orig_init(self, env, train_cfg, log_dir=log_dir, device=device)
        alg, runner, held = self.alg, self, {}

        def compute_returns(obs):
            del alg.compute_returns                         # one-shot: from here on the class's own method
            with torch.inference_mode(False):               # normal tensors, usable by the update later
                held["s0"] = snapshot(alg)
                held["obs"] = obs.clone()
            return alg.compute_returns(obs)

        def update(*a, **k):
            del alg.update
            if "s0" not in held:
                raise RuntimeError("update identity: update ran before compute_returns")
            mpath = os.path.join(tempfile.mkdtemp(prefix="sim2gate_update_identity_"), "signals.jsonl")
            doc = run_replays(runner, held["obs"], held["s0"], TrainingMonitor, mpath)
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(doc, f, indent=1)
            print(f"[sim2gate] update identity: {doc['status'].upper()}  (report: {out_path})", file=sys.stderr)
            with torch.inference_mode():                    # state is back at s0: finish this iteration as stock
                alg.compute_returns(held["obs"])
            held.clear()
            return alg.update(*a, **k)

        alg.compute_returns = compute_returns
        alg.update = update

    OnPolicyRunner.__init__ = __init__
    return lambda: setattr(OnPolicyRunner, "__init__", orig_init)
