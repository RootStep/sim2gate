"""TrainingMonitor: read-only logging of training-time signals from an RSL-RL OnPolicyRunner.

    runner = OnPolicyRunner(env, train_cfg, log_dir, device)
    monitor = TrainingMonitor(runner, "sim2gate_signals.jsonl")
    runner.learn(num_iterations)
    monitor.close()

Per PPO iteration it writes one JSON line with: same-critic BSRS signals (rho, regime, predicted rho,
zero-tail fraction; see signals.py), episode reward terms reported by the environment (Isaac Lab puts them in
extras["log"]["Episode_Reward/<term>"]), action std and learning rate. If the runner has a TensorBoard-style
writer, the main scalars also go there under "Sim2Gate/".

Read-only: the wrappers return exactly what the wrapped calls return, use no random numbers and change no
tensors that training uses. tests/test_monitor.py checks that final parameters are bit-identical with and
without the monitor. The one extra cost per iteration is a critic forward pass on the last observation.

For stock PPO, `eta` (default 1.0) is a what-if: the signals describe what same-critic BSRS at that eta would
do to this rollout's actor signal. For BSRSPPO the algorithm's own eta is used.
"""
import json
import torch
from .signals import rollout_signals

REWARD_PREFIX = "Episode_Reward/"


class TrainingMonitor:
    def __init__(self, runner, path, eta=None, reward_prefix=REWARD_PREFIX, writer=True):
        self.runner, self.alg = runner, runner.alg
        alg_eta = getattr(self.alg, "eta", None)
        self.eta = float(alg_eta if alg_eta is not None and alg_eta != 0.0 else (1.0 if eta is None else eta))
        self.measured = alg_eta is not None and alg_eta != 0.0
        self.reward_prefix = reward_prefix
        self.use_writer = writer
        self.f = open(path, "a", buffering=1)
        self.iteration = 0
        st = self.alg.storage
        self._time_outs = torch.zeros(st.num_transitions_per_env, st.num_envs, device=st.device, dtype=torch.bool)
        self._terms = {}
        self._pending = None
        self._restore = []
        self._install()

    # ---- wrappers -------------------------------------------------------------------------------
    def _install(self):
        alg, st = self.alg, self.alg.storage

        orig_pes = alg.process_env_step
        def process_env_step(obs, rewards, dones, extras):
            step = st.step
            to = extras.get("time_outs")
            if to is not None:
                self._time_outs[step].copy_(to.to(self._time_outs.device).reshape(-1).bool())
            else:
                self._time_outs[step].zero_()
            log = extras.get("log", extras.get("episode"))
            if isinstance(log, dict):
                for k, val in log.items():
                    if k.startswith(self.reward_prefix):
                        x = float(val.float().mean()) if torch.is_tensor(val) else float(val)
                        self._terms.setdefault(k[len(self.reward_prefix):], []).append(x)
            return orig_pes(obs, rewards, dones, extras)
        alg.process_env_step = process_env_step
        self._restore.append(lambda: setattr(alg, "process_env_step", orig_pes))

        orig_cr = alg.compute_returns
        def compute_returns(obs):
            with torch.inference_mode():
                last_values = alg.critic(obs).detach().clone()
                self._pending = dict(stored=st.rewards.clone(), values=st.values.clone(), dones=st.dones.clone(),
                                     time_outs=self._time_outs.clone(), last_values=last_values)
            return orig_cr(obs)
        alg.compute_returns = compute_returns
        self._restore.append(lambda: setattr(alg, "compute_returns", orig_cr))

        orig_update = alg.update
        def update(*a, **k):
            perms = []
            if alg.normalize_advantage_per_mini_batch:
                orig_randperm = torch.randperm
                def randperm(*ra, **rk):
                    out = orig_randperm(*ra, **rk); perms.append(out.clone()); return out
                torch.randperm = randperm
                try:
                    out = orig_update(*a, **k)
                finally:
                    torch.randperm = orig_randperm
            else:
                out = orig_update(*a, **k)
            self._log(out, perms)
            return out
        alg.update = update
        self._restore.append(lambda: setattr(alg, "update", orig_update))

    # ---- logging ----------------------------------------------------------------------------------
    def _log(self, loss_dict, perms):
        p = self._pending
        if p is None:
            return
        alg = self.alg
        mb_ix = None
        if perms:
            n = p["values"].shape[0] * p["values"].shape[1]
            size = n // alg.num_mini_batches
            mb_ix = [perms[0][i * size:(i + 1) * size] for i in range(alg.num_mini_batches)]
        with torch.no_grad():
            sig = rollout_signals(p["stored"], p["values"], p["last_values"], p["dones"], p["time_outs"],
                                  eta=self.eta, gamma=alg.gamma, lam=alg.lam, minibatch_indices=mb_ix)
        terms = {k: sum(v) / len(v) for k, v in self._terms.items() if v}
        total = sum(abs(x) for x in terms.values())
        shares = {k: (abs(x) / total if total > 0 else None) for k, x in terms.items()}
        try:
            std = float(alg.get_policy().output_std.detach().float().mean())
        except Exception:
            std = None
        rec = {"iteration": self.iteration, "eta": self.eta, "eta_is_training_eta": self.measured,
               "signals": sig.as_dict(), "reward_terms": terms, "reward_term_abs_share": shares,
               "action_std": std, "learning_rate": float(alg.learning_rate),
               "losses": {k: float(v) for k, v in (loss_dict or {}).items() if v is not None}}
        self.f.write(json.dumps(rec) + "\n")
        self._write_scalars(sig, shares)
        self._pending = None
        self._terms = {}
        self.iteration += 1

    def _write_scalars(self, sig, shares):
        if not self.use_writer:
            return
        writer = getattr(getattr(self.runner, "logger", None), "writer", None) or getattr(self.runner, "writer", None)
        if writer is None or not hasattr(writer, "add_scalar"):
            return
        it = getattr(self.runner, "current_learning_iteration", self.iteration)
        for k in ("rho", "predicted_rho", "zero_tail_fraction"):
            v = getattr(sig, k)
            if v is not None:
                writer.add_scalar(f"Sim2Gate/{k}", v, it)
        writer.add_scalar("Sim2Gate/sign_flip", 1.0 if sig.regime == "sign_flip" else 0.0, it)
        for k, v in shares.items():
            if v is not None:
                writer.add_scalar(f"Sim2Gate/reward_share/{k}", v, it)

    def close(self):
        for f in reversed(self._restore):
            f()
        self._restore = []
        self.f.close()
