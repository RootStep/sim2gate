"""VecMountainCar: a batched PyTorch port of MountainCarContinuous dynamics, used by the sim2gate tests (copied from the RSL-RL acceptance gate).

Implements rsl_rl.env.VecEnv the way Isaac Lab's RSL-RL wrapper does: observations are a TensorDict with a
"policy" group, dones = terminated | timed out (long), extras["time_outs"] flags pure time-limit truncations,
and the returned observation after a done is the reset observation.
"""
import math
import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv


class VecMountainCar(VecEnv):
    MIN_POS, MAX_POS, MAX_SPEED, GOAL = -1.2, 0.6, 0.07, 0.45
    POWER = 0.0015

    def __init__(self, num_envs=16, reward="sparse", max_episode_length=64, seed=0, device="cpu",
                 init_velocity=False):
        if reward not in ("sparse", "dense"):
            raise ValueError("reward must be sparse or dense")
        self.num_envs = num_envs
        self.num_actions = 1
        self.max_episode_length = max_episode_length
        self.device = device
        self.reward_mode = reward
        self.init_velocity = init_velocity   # addendum A: initial velocity uniform in [-0.07, 0.07]
        self.cfg = {"name": "VecMountainCar", "reward": reward, "max_episode_length": max_episode_length,
                    "init_velocity": init_velocity}
        self.gen = torch.Generator(device=device).manual_seed(10_000 + seed)
        self.pos = torch.zeros(num_envs, device=device)
        self.vel = torch.zeros(num_envs, device=device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._ep_goal = torch.zeros(num_envs, device=device)
        self._ep_effort = torch.zeros(num_envs, device=device)
        self._reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def _reset(self, mask):
        n = int(mask.sum())
        if n == 0:
            return
        start = torch.rand(n, generator=self.gen, device=self.device) * (0.44 + 0.6) - 0.6
        self.pos[mask] = start
        if self.init_velocity:
            self.vel[mask] = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * self.MAX_SPEED
        else:
            self.vel[mask] = 0.0
        self.episode_length_buf[mask] = 0

    def get_observations(self):
        obs = torch.stack([self.pos, self.vel * 10.0], dim=-1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions):
        a = actions.reshape(self.num_envs, -1)[:, 0].clamp(-1.0, 1.0)
        vel = self.vel + a * self.POWER - 0.0025 * torch.cos(3 * self.pos)
        vel = vel.clamp(-self.MAX_SPEED, self.MAX_SPEED)
        pos = (self.pos + vel).clamp(self.MIN_POS, self.MAX_POS)
        vel = torch.where((pos == self.MIN_POS) & (vel < 0), torch.zeros_like(vel), vel)
        self.pos, self.vel = pos, vel
        terminated = pos >= self.GOAL
        if self.reward_mode == "dense":
            rew = -0.1 * a * a + 100.0 * terminated.float()
        else:
            rew = terminated.float()
        self.episode_length_buf += 1
        time_outs = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        dones = terminated | time_outs
        self._reset(dones)
        extras = {"time_outs": time_outs.clone(), "terminated": terminated.clone()}
        # Isaac Lab style per-term episode logging, reported when episodes end
        self._ep_goal += terminated.float() * (100.0 if self.reward_mode == "dense" else 1.0)
        self._ep_effort += (-0.1 * a * a) if self.reward_mode == "dense" else torch.zeros_like(a)
        if bool(dones.any()):
            extras["log"] = {"Episode_Reward/goal": self._ep_goal[dones].mean(),
                             "Episode_Reward/effort": self._ep_effort[dones].mean()}
            self._ep_goal[dones] = 0.0
            self._ep_effort[dones] = 0.0
        return self.get_observations(), rew, dones.long(), extras
