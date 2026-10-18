"""EvidenceRecorder: the v2 phase 0A record of how each reward term shapes RSL-RL's PPO updates.

Built for Vihaan's question (v2 plan, phase 0B): can rho, defined for same-critic BSRS, be extended to the ordinary
reward terms of an Isaac Lab task? It records, without changing training:

every iteration (updates.jsonl, one line each)
    * each reward term's per-step contribution (Isaac Lab's RewardManager: term value * weight * dt), and how
      closely the terms add up to the reward the trainer received;
    * the moments (means and full covariance, float64) of [A, T_1 .. T_K, V], where A is the raw-reward GAE the
      trainer's critic gives, T_k is reward term k's discounted tail (GAE with V = 0) and V = GAE of zero reward
      (the critic's own part). GAE is linear in the rewards, so A = sum_k T_k + V exactly and removing term k
      gives exactly A - T_k: any correlation, norm ratio or regression of these quantities follows from the
      moments. The identity's residual is logged;
    * what-if same-critic BSRS rho at a few eta, the trainer residual (its own advantages vs the reconstruction),
      KL and clip fraction of the update on a fixed subset of environments, learning rate and losses.

deep iterations (deep/itNNNNN_*.npz|pt), chosen in advance
    * the policy-gradient direction g(A_hat) = grad_theta mean(A_hat * log pi(a|s)) over the whole rollout at the
      pre-update actor, for A_hat = the trainer's advantages, normalized A, normalized A - T_k for every term, and
      normalized same-critic BSRS A' at the what-if eta. PPO's surrogate gradient at ratio 1 is this direction;
      cosines with the trainer's are logged each deep iteration, vectors saved at the grad-save iterations;
    * the actor's loss gradient at every minibatch step of the update (after gradient clipping, as Adam gets it),
      and the actual change of the actor parameters made by the update (Adam, clipping, minibatches and all);
    * a sample of the rollout for the first `sample_envs` environments (observations, actions, old log-probs and
      distribution parameters, values, rewards per term, dones, time-outs, trainer advantages) and the pre-update
      actor and critic state_dicts, so any quantity can be recomputed offline;
    * optionally (refit iterations x refit terms): a copy of the critic refitted to the rewards without that term
      (fitted value iteration from the trainer's critic), the advantages it gives, and their gradient direction:
      the "critic issue" of v2 item 6, measured instead of assumed.

Read-only: every computation runs on copies or under torch.random.fork_rng and never writes a tensor training uses;
tests/test_evidence.py checks that final parameters are bit-identical with and without the recorder. Feed-forward
actor and critic only; no RND, no symmetry augmentation.
"""
import copy
import importlib.metadata as metadata
import json
import os

import numpy as np
import torch

from .signals import correlation, gae, reward_tail, shaped_rewards

FORMAT_VERSION = 1
DEFAULT_DEEP = (0, 1, 2, 5, 10, 20, 50) + tuple(range(100, 2000, 100)) + (1999,)
DEFAULT_GRAD_SAVE = (0, 10, 50, 100, 200, 500, 1000, 1500, 1999)
DEFAULT_WHAT_IF_ETAS = (0.5, 1.0, 2.0)


class IsaacLabRewardTerms:
    """Per-step reward terms from an Isaac Lab ManagerBasedRLEnv (through RSL-RL's wrapper).

    Isaac Lab's RewardManager.compute(dt) adds term_value * weight * dt to the reward for each term and stores
    term_value * weight in _step_reward; it is not cleared on reset, so after env.step() it holds that step's terms.
    The per-step contribution is therefore _step_reward * step_dt, and the terms add up to the step's reward."""

    def __init__(self, env):
        u = env
        for _ in range(4):
            if hasattr(u, "reward_manager"):
                break
            u = getattr(u, "unwrapped", None) or getattr(u, "env", None)
            if u is None:
                break
        if u is None or not hasattr(u, "reward_manager"):
            raise ValueError("no Isaac Lab reward manager found on the environment")
        self.rm = u.reward_manager
        self.names = list(self.rm.active_terms)
        self.dt = float(u.step_dt)

    def step_terms(self):
        return self.rm._step_reward * self.dt


def _flat_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def _moments(cols):
    """Means and covariance (float64, ddof 1) of the columns of a [n, m] matrix."""
    x = cols.double()
    mean = x.mean(0)
    xc = x - mean
    cov = xc.T @ xc / max(x.shape[0] - 1, 1)
    return mean, cov


def _corr_from_cov(cov, i, w):
    """corr(column i, sum_j w_j column j) from a covariance matrix."""
    w = w.double()
    var_i = cov[i, i]
    var_w = w @ cov @ w
    c = cov[i] @ w
    den = (var_i * var_w).sqrt()
    return None if float(den) == 0.0 else float((c / den).clamp(-1, 1))


def _normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


class EvidenceRecorder:
    def __init__(self, runner, out_dir, terms=None, deep=DEFAULT_DEEP, grad_save=DEFAULT_GRAD_SAVE,
                 sample_envs=256, what_if_etas=DEFAULT_WHAT_IF_ETAS, refit_terms=(), refit_iterations=(),
                 refit_rounds=3, refit_epochs=5, refit_lr=1e-3, meta=None):
        self.runner, self.alg = runner, runner.alg
        alg, st = self.alg, self.alg.storage
        for m in (getattr(alg, "actor", None), getattr(alg, "critic", None)):
            if getattr(m, "is_recurrent", False):
                raise NotImplementedError("EvidenceRecorder supports feed-forward actor and critic only")
        if getattr(alg, "rnd", None) or getattr(alg, "symmetry", None):
            raise NotImplementedError("EvidenceRecorder does not support RND or symmetry augmentation")
        self.terms = terms or IsaacLabRewardTerms(runner.env)
        self.names = list(self.terms.names)
        unknown = [t for t in refit_terms if t not in self.names]
        if unknown:
            raise ValueError(f"refit terms {unknown} are not reward terms ({self.names})")
        self.out = out_dir
        self.deep_dir = os.path.join(out_dir, "deep")
        os.makedirs(self.deep_dir, exist_ok=True)
        self.deep, self.grad_save = set(int(i) for i in deep), set(int(i) for i in grad_save)
        self.refit_terms, self.refit_iterations = tuple(refit_terms), set(int(i) for i in refit_iterations)
        self.refit_rounds, self.refit_epochs, self.refit_lr = int(refit_rounds), int(refit_epochs), float(refit_lr)
        self.what_if_etas = tuple(float(e) for e in what_if_etas)
        T, N, K = st.num_transitions_per_env, st.num_envs, len(self.names)
        self.S = min(int(sample_envs), N)
        self.device = st.device
        self._terms = torch.zeros(T, N, K, device=self.device)
        self._raw = torch.zeros(T, N, device=self.device)
        self._to = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self._term_sum_res = 0.0
        self._pending = None
        self.iteration = 0
        self._restore = []
        self.f = open(os.path.join(out_dir, "updates.jsonl"), "a", buffering=1)
        self._write_meta(meta or {})
        self._install()

    # ---- metadata -----------------------------------------------------------------------------------------------
    def _write_meta(self, extra):
        alg, st = self.alg, self.alg.storage

        def version(pkg):
            try:
                return metadata.version(pkg)
            except metadata.PackageNotFoundError:
                return None
        meta = {
            "format_version": FORMAT_VERSION, "sim2gate": version("sim2gate"), "rsl_rl_lib": version("rsl-rl-lib"),
            "torch": torch.__version__, "algorithm": type(alg).__name__, "eta": float(getattr(alg, "eta", 0.0) or 0.0),
            "gamma": float(alg.gamma), "lam": float(alg.lam), "clip_param": float(alg.clip_param),
            "num_learning_epochs": int(alg.num_learning_epochs), "num_mini_batches": int(alg.num_mini_batches),
            "normalize_advantage_per_mini_batch": bool(alg.normalize_advantage_per_mini_batch),
            "num_envs": int(st.num_envs), "num_steps_per_env": int(st.num_transitions_per_env),
            "reward_terms": self.names, "step_dt": getattr(self.terms, "dt", None), "sample_envs": self.S,
            "deep_iterations": sorted(self.deep), "grad_save_iterations": sorted(self.grad_save),
            "what_if_etas": list(self.what_if_etas),
            "refit": {"terms": list(self.refit_terms), "iterations": sorted(self.refit_iterations),
                      "rounds": self.refit_rounds, "epochs": self.refit_epochs, "lr": self.refit_lr},
            "actor": repr(alg.actor), "critic": repr(alg.critic),
            "actor_parameters": int(sum(p.numel() for p in alg.actor.parameters())),
            "moments_columns": ["A"] + [f"T:{n}" for n in self.names] + ["V"],
            "notes": "A: raw-reward GAE with the trainer's critic and RSL-RL's time-out bootstrap; T:k: GAE of term k "
                     "with V = 0 (its discounted tail); V: GAE of zero reward. A = sum_k T:k + V. Gradient directions "
                     "are ascent directions grad mean(A_hat * log pi); PPO's step descends -that.",
        }
        meta.update(extra)
        with open(os.path.join(self.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    # ---- wrappers -----------------------------------------------------------------------------------------------
    def _install(self):
        alg, st = self.alg, self.alg.storage

        orig_pes = alg.process_env_step

        def process_env_step(obs, rewards, dones, extras):
            step = st.step
            with torch.no_grad():
                t = self.terms.step_terms().detach().to(self._terms.dtype).to(self.device)
                self._terms[step].copy_(t)
                r = rewards.detach().reshape(-1).to(self._raw.dtype).to(self.device)
                self._raw[step].copy_(r)
                self._term_sum_res = max(self._term_sum_res, float((t.sum(-1) - r).abs().max()))
                to = extras.get("time_outs")
                if to is None:
                    self._to[step].zero_()
                else:
                    self._to[step].copy_(to.to(self.device).reshape(-1).bool())
            return orig_pes(obs, rewards, dones, extras)
        alg.process_env_step = process_env_step
        self._restore.append(lambda: setattr(alg, "process_env_step", orig_pes))

        orig_cr = alg.compute_returns

        def compute_returns(obs):
            with torch.no_grad():
                last_values = alg.critic(obs).detach().clone()
            p = dict(values=st.values.detach().clone(), dones=st.dones.detach().clone(), time_outs=self._to.clone(),
                     raw=self._raw.clone(), terms=self._terms.clone(), stored=st.rewards.detach().clone(),
                     last_values=last_values, last_obs=obs.clone() if hasattr(obs, "clone") else obs,
                     term_sum_residual=self._term_sum_res)
            self._term_sum_res = 0.0
            out = orig_cr(obs)
            p["trainer_adv"] = (st.returns - st.values).detach().clone()
            p["trainer_norm"] = None if alg.normalize_advantage_per_mini_batch else st.advantages.detach().clone()
            self._pending = p
            return out
        alg.compute_returns = compute_returns
        self._restore.append(lambda: setattr(alg, "compute_returns", orig_cr))

        orig_update = alg.update

        def update(*a, **k):
            pre = self._before_update() if self._pending is not None else None
            undo = self._capture_minibatch_grads(pre) if pre is not None and "dirs" in pre else None
            try:
                out = orig_update(*a, **k)
            finally:
                if undo:
                    undo()
            if pre is not None:
                self._after_update(pre, out)
            return out
        alg.update = update
        self._restore.append(lambda: setattr(alg, "update", orig_update))

    # ---- computations -------------------------------------------------------------------------------------------
    def _sq(self, x):
        return x[..., 0] if x.dim() == 3 and x.shape[-1] == 1 else x

    def _advantage_parts(self, p, values=None, last_values=None):
        alg = self.alg
        v = self._sq(p["values"]) if values is None else values
        lv = (p["last_values"] if last_values is None else last_values).reshape(-1).to(v.dtype)
        d, to, raw = self._sq(p["dones"]).bool(), p["time_outs"], p["raw"].to(v.dtype)
        A = gae(raw, v, lv, d, to, gamma=alg.gamma, lam=alg.lam)
        Vp = gae(torch.zeros_like(raw), v, lv, d, to, gamma=alg.gamma, lam=alg.lam)
        Tk = torch.stack([reward_tail(p["terms"][..., k].to(v.dtype), d, gamma=alg.gamma, lam=alg.lam)
                          for k in range(len(self.names))], dim=-1)
        Tall = reward_tail(raw, d, gamma=alg.gamma, lam=alg.lam)
        return A, Tk, Vp, Tall

    def _actor_logp(self, obs, actions):
        """log pi(a|s) for stored samples at the current actor, with grad, drawing no random numbers."""
        actor = self.alg.actor
        latent = actor.get_latent(obs)
        actor.distribution.update(actor.mlp(latent))
        return actor.get_output_log_prob(actions).reshape(-1)

    def _actor_params(self):
        return [q for q in self.alg.actor.parameters() if q.requires_grad]

    def _directions(self, variants):
        """{name: ascent direction} over the whole rollout at the current actor."""
        st = self.alg.storage
        obs = st.observations.flatten(0, 1).clone()
        actions = st.actions.flatten(0, 1).clone()
        params = self._actor_params()
        with torch.enable_grad():
            logp = self._actor_logp(obs, actions)
            out = {}
            names = list(variants)
            for i, name in enumerate(names):
                a_hat = variants[name].reshape(-1).to(logp.dtype).detach()
                g = torch.autograd.grad((a_hat * logp).mean(), params, retain_graph=i < len(names) - 1)
                out[name] = torch.cat([x.reshape(-1) for x in g]).detach()
        return out

    def _refit(self, p, term):
        """A copy of the critic refitted to the rewards without `term` (fitted value iteration, warm start)."""
        alg, st = self.alg, self.alg.storage
        k = self.names.index(term)
        crit = copy.deepcopy(alg.critic)
        opt = torch.optim.Adam([q for q in crit.parameters() if q.requires_grad], lr=self.refit_lr)
        obs = st.observations.flatten(0, 1).clone()
        T, N = st.num_transitions_per_env, st.num_envs
        d, to = self._sq(p["dones"]).bool(), p["time_outs"]
        r = (p["raw"] - p["terms"][..., k]).to(torch.float32)
        losses = []

        def values():
            with torch.no_grad():
                return crit(obs).reshape(T, N), crit(p["last_obs"]).reshape(-1)
        for _ in range(self.refit_rounds):
            v, lv = values()
            target = (gae(r, v, lv, d, to, gamma=alg.gamma, lam=alg.lam) + v).reshape(-1).detach()
            n = target.numel()
            mb = max(n // max(alg.num_mini_batches, 1), 1)
            for _e in range(self.refit_epochs):
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n - mb + 1, mb):
                    ix = perm[i:i + mb]
                    loss = (crit(obs[ix]).reshape(-1) - target[ix]).pow(2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                losses.append(float((crit(obs).reshape(-1) - target).pow(2).mean()))
        v, lv = values()
        A_refit = gae(r, v, lv, d, to, gamma=alg.gamma, lam=alg.lam)
        return A_refit, losses

    def _before_update(self):
        """Everything that needs the pre-update networks: rollout statistics, directions, sample dump, refits."""
        alg, st, it = self.alg, self.alg.storage, self.iteration
        # the rollout runs under torch.inference_mode: clone into ordinary tensors so autograd can use them
        p = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in self._pending.items()}
        rec = {"iteration": it, "deep": it in self.deep}
        with torch.random.fork_rng(devices=[self.device] if str(self.device).startswith("cuda") else []):
            with torch.no_grad():
                A, Tk, Vp, Tall = self._advantage_parts(p)
                K = len(self.names)
                cols = torch.cat([A.reshape(-1, 1), Tk.reshape(-1, K), Vp.reshape(-1, 1)], dim=1)
                mean, cov = _moments(cols)
                ident = float((A.double() - (Tk.double().sum(-1) + Vp.double())).abs().max())
                terms = p["terms"].double()
                term_mean = terms.mean((0, 1))
                term_abs = terms.abs().mean((0, 1))
                eta = float(getattr(alg, "eta", 0.0) or 0.0)
                d, to = self._sq(p["dones"]).bool(), p["time_outs"]
                v, lv = self._sq(p["values"]), p["last_values"].reshape(-1)
                trained = (gae(shaped_rewards(p["raw"].to(v.dtype), v, lv, d, to, eta=eta, gamma=alg.gamma),
                               v, lv, d, to, gamma=alg.gamma, lam=alg.lam) if eta != 0.0 else A)
                tr = self._sq(p["trainer_adv"]).double()
                rec.update({
                    "term_step_mean": term_mean.tolist(), "term_step_abs_mean": term_abs.tolist(),
                    "term_abs_share": (term_abs / term_abs.sum()).tolist() if float(term_abs.sum()) > 0 else None,
                    "term_sum_residual": p["term_sum_residual"], "identity_residual": ident,
                    "trainer_residual": float((tr - trained.double()).abs().max() / tr.abs().max().clamp_min(1.0)),
                    "moments": {"mean": mean.tolist(), "cov": cov.tolist()},
                    "corr_A_A_minus_term": [_corr_from_cov(cov, 0, torch.eye(K + 2, dtype=torch.float64)[0]
                                                           - torch.eye(K + 2, dtype=torch.float64)[1 + k])
                                            for k in range(K)],
                    "bsrs_what_if_rho": {f"{e:g}": _corr_from_cov(
                        _moments(torch.stack([A.reshape(-1), Tall.reshape(-1)], 1))[1], 0,
                        torch.tensor([1.0 + e, -e], dtype=torch.float64)) for e in self.what_if_etas},
                    "timeouts": int(to.sum()), "terminations": int((d & ~to).sum()),
                })
            pre = {"rec": rec, "A": A, "Tk": Tk, "Tall": Tall}
            if rec["deep"] or it in self.refit_iterations:
                with torch.no_grad():
                    variants = {"trainer": (self._sq(p["trainer_norm"]) if p["trainer_norm"] is not None
                                            else _normalize(self._sq(p["trainer_adv"]))),
                                "A": _normalize(A)}
                    for k, n in enumerate(self.names):
                        variants[f"minus:{n}"] = _normalize(A - Tk[..., k])
                    for e in self.what_if_etas:
                        variants[f"bsrs:{e:g}"] = _normalize((1 + e) * A - e * Tall)
                refit = {}
                if it in self.refit_iterations:
                    for term in self.refit_terms:
                        A_r, losses = self._refit(p, term)
                        k = self.names.index(term)
                        variants[f"refit_minus:{term}"] = _normalize(A_r)
                        refit[term] = {"value_loss_by_round": losses,
                                       "corr_A": _c(A, A_r), "corr_A_minus_term": _c(A - Tk[..., k], A_r)}
                    rec["refit"] = refit
                dirs = self._directions(variants)
                g0 = dirs["trainer"]
                rec["direction_norm"] = {n: float(g.norm()) for n, g in dirs.items()}
                rec["direction_cos_trainer"] = {n: _cos(g0, g) for n, g in dirs.items()}
                pre["dirs"] = dirs
                pre["theta0"] = _flat_params(self._actor_params()).clone()
                if rec["deep"]:
                    self._dump_samples(p, it)
        return pre

    def _capture_minibatch_grads(self, pre):
        """Record the actor's gradient at every optimizer step of this update (after RSL-RL's gradient clipping,
        i.e. exactly what Adam receives). Reads .grad only."""
        opt, params = self.alg.optimizer, self._actor_params()
        orig = opt.step
        pre["minibatch_grads"] = grads = []

        def step(*a, **k):
            grads.append(torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).detach().reshape(-1)
                                    for q in params]).clone())
            return orig(*a, **k)
        opt.step = step
        return lambda: opt.__dict__.pop("step", None)

    def _dump_samples(self, p, it):
        alg, st, S = self.alg, self.alg.storage, self.S
        arr = lambda x: x.detach().float().cpu().numpy()  # noqa: E731
        data = {}
        for key in st.observations.keys():
            data[f"obs_{key}"] = arr(st.observations[key][:, :S])
        data["actions"] = arr(st.actions[:, :S])
        data["old_log_prob"] = arr(self._sq(st.actions_log_prob[:, :S]))
        for i, q in enumerate(st.distribution_params):
            data[f"old_dist_param_{i}"] = arr(q[:, :S])
        data["values"] = arr(self._sq(p["values"][:, :S]))
        data["last_values"] = arr(p["last_values"].reshape(-1)[:S])
        data["stored_rewards"] = arr(self._sq(p["stored"][:, :S]))
        data["raw_rewards"] = arr(p["raw"][:, :S])
        data["terms"] = arr(p["terms"][:, :S])
        data["dones"] = arr(self._sq(p["dones"][:, :S]))
        data["time_outs"] = arr(p["time_outs"][:, :S])
        data["trainer_advantages_unnormalized"] = arr(self._sq(p["trainer_adv"][:, :S]))
        if p["trainer_norm"] is not None:
            data["trainer_advantages"] = arr(self._sq(p["trainer_norm"][:, :S]))
        np.savez_compressed(os.path.join(self.deep_dir, f"it{it:05d}_samples.npz"),
                            term_names=np.array(self.names), **data)
        torch.save({"actor": alg.actor.state_dict(), "critic": alg.critic.state_dict()},
                   os.path.join(self.deep_dir, f"it{it:05d}_models.pt"))

    def _after_update(self, pre, loss_dict):
        alg, st, it, rec = self.alg, self.alg.storage, self.iteration, pre["rec"]
        S = self.S
        with torch.random.fork_rng(devices=[self.device] if str(self.device).startswith("cuda") else []):
            with torch.no_grad():
                obs = st.observations[:, :S].flatten(0, 1).clone()
                actions = st.actions[:, :S].flatten(0, 1).clone()
                old = tuple(q[:, :S].flatten(0, 1) for q in st.distribution_params)
                logp = self._actor_logp(obs, actions)
                new = tuple(q.detach() for q in alg.actor.output_distribution_params)
                kl = alg.actor.get_kl_divergence(old, new)
                ratio = torch.exp(logp - st.actions_log_prob[:, :S].reshape(-1))
                rec["kl_subset"] = float(kl.mean())
                rec["clip_fraction_subset"] = float(((ratio - 1.0).abs() > alg.clip_param).double().mean())
                if "dirs" in pre:
                    delta = _flat_params(self._actor_params()) - pre["theta0"]
                    rec["update_norm"] = float(delta.norm())
                    rec["update_cos"] = {n: _cos(g, delta) for n, g in pre["dirs"].items()}
                    mg = pre.get("minibatch_grads") or []
                    # loss gradients; their negatives are ascent directions comparable with the directions above
                    rec["minibatch_grad_norm"] = [float(g.norm()) for g in mg]
                    rec["minibatch_grad_cos_trainer"] = [_cos(-g, pre["dirs"]["trainer"]) for g in mg]
                    if it in self.grad_save:
                        names = list(pre["dirs"])
                        np.savez_compressed(
                            os.path.join(self.deep_dir, f"it{it:05d}_directions.npz"),
                            names=np.array(names),
                            directions=torch.stack([pre["dirs"][n] for n in names]).float().cpu().numpy(),
                            actual_update=delta.float().cpu().numpy(),
                            minibatch_loss_grads=(torch.stack(mg).float().cpu().numpy() if mg
                                                  else np.zeros((0, delta.numel()), np.float32)),
                            param_shapes=np.array([str(tuple(q.shape)) for q in self._actor_params()]))
        rec["learning_rate"] = float(alg.learning_rate)
        rec["losses"] = {k: float(v) for k, v in (loss_dict or {}).items() if v is not None}
        self.f.write(json.dumps(rec) + "\n")
        self._pending = None
        self.iteration += 1

    def close(self):
        for f in reversed(self._restore):
            f()
        self._restore = []
        self.f.close()


def _cos(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    den = a.norm() * b.norm()
    return None if float(den) == 0.0 else float((a @ b / den).clamp(-1, 1))


def _c(a, b):
    return correlation(a.reshape(-1), b.reshape(-1))
