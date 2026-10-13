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

Every record also checks the reconstruction against what the trainer actually computed (trainer_signal_residual:
its own returns - values vs the reconstructed A' for BSRSPPO, or A for stock PPO), so a trainer that does not
apply the claimed shaping shows up. Feed-forward actor and critic only.

When the trainer normalizes advantages per minibatch, the monitor wraps the storage's own mini_batch_generator:
it records the one shuffle that generator draws and the advantage tensors it actually yields, checks that each
yielded tensor is exactly the trainer's advantages at the recorded indices, and computes the per-minibatch
statistics on those groups. If any of that cannot be established, the record keeps normalization_group
"minibatch", reports the reason in minibatch_capture and leaves the group statistics empty; it never falls back to
rollout statistics.

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
        for m in (getattr(self.alg, "actor", None), getattr(self.alg, "critic", None)):
            if getattr(m, "is_recurrent", False):
                raise NotImplementedError("TrainingMonitor supports feed-forward actor/critic only: its extra critic "
                                          "forward pass would advance a recurrent hidden state")
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
        self._mb = None
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
            out = orig_cr(obs)
            # what the trainer actually produced: unnormalized advantages, and what it will train on
            self._pending["trainer_adv"] = (st.returns - st.values).detach().clone()
            self._pending["trainer_norm"] = (None if alg.normalize_advantage_per_mini_batch
                                             else st.advantages.detach().clone())
            return out
        alg.compute_returns = compute_returns
        self._restore.append(lambda: setattr(alg, "compute_returns", orig_cr))

        if alg.normalize_advantage_per_mini_batch:
            had_own = "mini_batch_generator" in vars(st)
            orig_gen = st.mini_batch_generator

            def mini_batch_generator(num_mini_batches, *ga, **gk):
                """Pass-through: same batches, same order, same random numbers; records the shuffle the
                generator draws and the advantages it yields for the first epoch."""
                cap = {"perms": [], "yielded": [], "num_mini_batches": int(num_mini_batches)}
                self._mb = cap
                g = orig_gen(num_mini_batches, *ga, **gk)
                orig_randperm = torch.randperm
                def randperm(*ra, **rk):
                    out = orig_randperm(*ra, **rk); cap["perms"].append(out.clone()); return out
                torch.randperm = randperm          # the generator body runs up to its first yield inside next()
                try:
                    batch = next(g)
                except StopIteration:
                    return
                finally:
                    torch.randperm = orig_randperm
                while True:
                    if len(cap["yielded"]) < cap["num_mini_batches"]:
                        cap["yielded"].append(batch.advantages.detach().reshape(-1).clone())
                    yield batch
                    try:
                        batch = next(g)
                    except StopIteration:
                        return
            st.mini_batch_generator = mini_batch_generator
            self._restore.append(lambda: delattr(st, "mini_batch_generator") if not had_own
                                 else setattr(st, "mini_batch_generator", orig_gen))

        orig_update = alg.update
        def update(*a, **k):
            self._mb = None
            out = orig_update(*a, **k)
            self._log(out)
            return out
        alg.update = update
        self._restore.append(lambda: setattr(alg, "update", orig_update))

    # ---- logging ----------------------------------------------------------------------------------
    def _minibatch_groups(self, p):
        """(indices, yielded advantages, None) for the update's minibatches, or (None, None, reason)."""
        cap = self._mb
        if cap is None:
            return None, None, "the storage's mini_batch_generator was not used in this update"
        m = cap["num_mini_batches"]
        n = p["values"].shape[0] * p["values"].shape[1]
        size = n // m
        if len(cap["perms"]) != 1 or cap["perms"][0].numel() != m * size:
            return None, None, (f"expected the generator to draw one shuffle of {m * size} indices, saw "
                                f"{[int(q.numel()) for q in cap['perms']]}")
        if len(cap["yielded"]) != m:
            return None, None, f"expected {m} minibatches in the first epoch, saw {len(cap['yielded'])}"
        perm = cap["perms"][0].to(p["trainer_adv"].device)
        ix = [perm[i * size:(i + 1) * size] for i in range(m)]
        flat = p["trainer_adv"].reshape(-1)
        for i, (j, y) in enumerate(zip(ix, cap["yielded"])):
            if y.numel() != j.numel() or not torch.equal(y.to(flat.dtype).to(flat.device), flat[j]):
                return None, None, f"minibatch {i}: yielded advantages are not the trainer's at the recorded indices"
        return ix, cap["yielded"], None

    def _log(self, loss_dict):
        p = self._pending
        if p is None:
            return
        alg = self.alg
        mb_ix, yielded, group, capture = None, None, "rollout", None
        if alg.normalize_advantage_per_mini_batch:
            group = "minibatch"
            mb_ix, yielded, reason = self._minibatch_groups(p)
            capture = "ok" if reason is None else reason
        with torch.no_grad():
            sig = rollout_signals(p["stored"], p["values"], p["last_values"], p["dones"], p["time_outs"],
                                  eta=self.eta, gamma=alg.gamma, lam=alg.lam, minibatch_indices=mb_ix,
                                  trainer_advantages=p.get("trainer_adv"), trainer_normalized=p.get("trainer_norm"),
                                  training_eta=self.measured, minibatch_capture=capture,
                                  normalization_group=group, minibatch_yielded=yielded)
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
        self._mb = None
        self._terms = {}
        self.iteration += 1

    def _write_scalars(self, sig, shares):
        if not self.use_writer:
            return
        writer = getattr(getattr(self.runner, "logger", None), "writer", None) or getattr(self.runner, "writer", None)
        if writer is None or not hasattr(writer, "add_scalar"):
            return
        it = getattr(self.runner, "current_learning_iteration", self.iteration)
        groups = [g for g in (sig.groups or []) if g["rho"] is not None]
        if groups:   # statistics over the groups RSL-RL actually normalizes (rollout, or each minibatch)
            writer.add_scalar("Sim2Gate/rho", sum(g["rho"] for g in groups) / len(groups), it)
            writer.add_scalar("Sim2Gate/sign_flip_share",
                              sum(g["regime"] == "sign_flip" for g in groups) / len(groups), it)
        if sig.predicted_rho is not None:
            writer.add_scalar("Sim2Gate/rollout_predicted_rho", sig.predicted_rho, it)
        writer.add_scalar("Sim2Gate/zero_tail_fraction", sig.zero_tail_fraction, it)
        if sig.trainer_signal_residual is not None:
            writer.add_scalar("Sim2Gate/trainer_signal_residual", sig.trainer_signal_residual, it)
        for k, v in shares.items():
            if v is not None:
                writer.add_scalar(f"Sim2Gate/reward_share/{k}", v, it)

    def close(self):
        for f in reversed(self._restore):
            f()
        self._restore = []
        self.f.close()
