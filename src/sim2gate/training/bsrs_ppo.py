"""BSRSPPO: RSL-RL's PPO with same-critic BSRS reward shaping, and nothing else changed.

    r'_t = r_t + eta * (gamma * b_t * V_{t+1} - V_t)

using RSL-RL's own conventions: the stored reward already contains gamma * V(s_t) on time-limit steps, the
successor potential on a timeout is V(s_t) (the value RSL-RL bootstraps with), it is 0 on a true termination,
values[t+1] inside the rollout and the critic's value of the final observation at the rollout's end.
Returns and advantages are then computed by RSL-RL's own recursion on the shaped rewards, and normalized
exactly as stock PPO does. With eta = 0 every call goes straight to PPO.
"""
import torch
from rsl_rl.algorithms import PPO


class BSRSPPO(PPO):
    def __init__(self, *args, eta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = float(eta)
        self._time_outs = None
        self.last_raw_advantages = None   # raw-reward GAE from RSL-RL's recursion, before shaping

    def process_env_step(self, obs, rewards, dones, extras):
        if self.eta != 0.0:
            st = self.storage
            if self._time_outs is None:
                self._time_outs = torch.zeros(st.num_transitions_per_env, st.num_envs, 1, device=self.device)
            to = extras.get("time_outs")
            to = torch.zeros_like(rewards) if to is None else to
            self._time_outs[st.step].copy_(to.to(self.device).view(-1, 1).float())
        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs):
        if self.eta == 0.0:
            return super().compute_returns(obs)
        st = self.storage
        stored = st.rewards.clone()
        # 1) raw-reward GAE with RSL-RL's own recursion, normalization skipped
        keep = self.normalize_advantage_per_mini_batch
        self.normalize_advantage_per_mini_batch = True
        try:
            super().compute_returns(obs)
        finally:
            self.normalize_advantage_per_mini_batch = keep
        self.last_raw_advantages = (st.returns - st.values).clone()
        # 2) same-critic shaping
        last_values = self.critic(obs).detach()
        next_values = torch.cat([st.values[1:], last_values.unsqueeze(0)], dim=0)
        to = self._time_outs.bool()
        term = st.dones.bool() & ~to
        phi_next = torch.where(to, st.values, torch.where(term, torch.zeros_like(next_values), next_values))
        st.rewards.copy_(stored + self.eta * (self.gamma * phi_next - st.values))
        # 3) shaped returns and advantages, normalized as stock PPO does
        super().compute_returns(obs)
        st.rewards.copy_(stored)
