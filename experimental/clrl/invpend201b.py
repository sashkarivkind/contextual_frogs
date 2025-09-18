#!/usr/bin/env python3
"""
Batched Vanilla Policy Gradient (REINFORCE) for a continuous-action Inverted Pendulum.

- In-house batched environment (no Gym).
- Gaussian policy with 1-D mean output; fixed std.
- Default batch_size=64; mean return is computed across the batch.
- Episode loop collects one episode per env in parallel (variable lengths handled).

Outputs:
  - learning_curve_batched_ip.png : mean return per episode (averaged across batch)

Author: you + ChatGPT
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =========================================
# In-house Batched Inverted Pendulum Env
# =========================================

@dataclass
class InvertedPendulumConfig:
    gravity: float = 9.81     # m/s^2
    mass: float = 1.0         # kg
    length: float = 1.0       # m (point-mass at end of rod)
    damping: float = 0.05     # viscous friction (N*m*s/rad)
    max_torque: float = 2.0   # |u| ≤ max_torque
    tau: float = 0.02         # s (integration step)
    angle_threshold: float = math.radians(24.0)  # terminate if |theta| > threshold
    max_steps: int = 400      # episode cap

    # initial state noise around upright
    init_theta_low: float = math.radians(-5.0)
    init_theta_high: float = math.radians(+5.0)
    init_dtheta_low: float = -0.5
    init_dtheta_high: float = +0.5


class BatchedInvertedPendulumEnv:
    """
    Batched inverted pendulum. Tracks 'done' internally so you can keep calling step()
    until all environments finish. After an env is done, it returns reward=0 and keeps
    its terminal state.
    State per env: [theta, theta_dot], theta=0 is upright.
    Action per env: scalar torque u in [-max_torque, max_torque].
    Reward: +1 per alive step, else 0 after done.
    """
    def __init__(self, batch_size: int = 64, seed: int | None = None, config: InvertedPendulumConfig | None = None):
        self.cfg = config if config is not None else InvertedPendulumConfig()
        self.batch_size = int(batch_size)
        # per-env RNGs for reproducibility
        base_seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.np_randoms = [np.random.RandomState(base_seed + i * 9973) for i in range(self.batch_size)]
        self.state = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.steps = np.zeros((self.batch_size,), dtype=np.int32)
        self.done = np.zeros((self.batch_size,), dtype=bool)
        self._update_derived()

    def _update_derived(self):
        self.I = self.cfg.mass * (self.cfg.length ** 2)  # inertia for point mass at end

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        return (2,)  # [theta, theta_dot]

    def set_config(self, new_cfg: InvertedPendulumConfig):
        self.cfg = new_cfg
        self._update_derived()

    def reset(self) -> np.ndarray:
        th = np.array([rng.uniform(self.cfg.init_theta_low, self.cfg.init_theta_high)
                       for rng in self.np_randoms], dtype=np.float32)
        dth = np.array([rng.uniform(self.cfg.init_dtheta_low, self.cfg.init_dtheta_high)
                        for rng in self.np_randoms], dtype=np.float32)
        self.state[:, 0] = th
        self.state[:, 1] = dth
        self.steps[:] = 0
        self.done[:] = False
        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        actions: np.ndarray shape [B] (float)
        returns:
            obs:    [B,2] float32
            reward: [B] float32
            done:   [B] bool
        """
        actions = np.asarray(actions, dtype=np.float32).reshape(-1)
        assert actions.shape[0] == self.batch_size, "actions must be shape [batch_size]"

        # Default: zeros
        rewards = np.zeros((self.batch_size,), dtype=np.float32)

        # Only step environments that are not done
        active = ~self.done
        if np.any(active):
            theta = self.state[active, 0].astype(np.float64)
            theta_dot = self.state[active, 1].astype(np.float64)

            # clamp actions to env bounds
            u = np.clip(actions[active], -self.cfg.max_torque, self.cfg.max_torque).astype(np.float64)

            # dynamics: I*theta_ddot = m*g*L*sin(theta) + u - b*theta_dot
            theta_ddot = (self.cfg.mass * self.cfg.gravity * self.cfg.length * np.sin(theta)
                          + u - self.cfg.damping * theta_dot) / (self.I + 1e-8)

            # symplectic Euler
            theta_dot = theta_dot + self.cfg.tau * theta_ddot
            theta = theta + self.cfg.tau * theta_dot

            # wrap angles to [-pi, pi] (optional but helps numerics)
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            # write back
            self.state[active, 0] = theta.astype(np.float32)
            self.state[active, 1] = theta_dot.astype(np.float32)

            # rewards for active envs that are still alive this step
            rewards[active] = 1.0

            # step counters
            self.steps[active] += 1

            # termination
            term_angle = np.abs(self.state[active, 0]) > self.cfg.angle_threshold
            term_steps = self.steps[active] >= self.cfg.max_steps
            newly_done = np.zeros_like(active)
            newly_done[active] = np.logical_or(term_angle, term_steps)
            self.done = np.logical_or(self.done, newly_done)

        return self._get_obs(), rewards, self.done.copy(), {"clipped_actions": actions}

    def _get_obs(self) -> np.ndarray:
        return self.state.copy()


# =========================================
# Policy: Gaussian mean, fixed std
# =========================================

class GaussianPolicyMean(nn.Module):
    """
    Outputs the torque mean (1-D). Fixed std is provided externally.
    """
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_head = nn.Linear(hidden, 1)
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.mean_head.weight, nonlinearity="linear")
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.mean_head(x)  # [B,1]


# =========================================
# Training utilities
# =========================================

@dataclass
class TrainConfig:
    episodes: int = 300
    batch_size: int = 64              # <— default as requested
    gamma: float = 0.99
    lr: float = 1e-1
    hidden_size: int = 128
    grad_clip: float = 1.0
    fixed_std: float = 0.5
    entropy_coef: float = 1e-3
    print_every: int = 10
    seed: int = 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_returns(rewards: List[float], gamma: float, device: torch.device) -> torch.Tensor:
    """
    Reward-to-go for a single trajectory (list of floats) — centered only (no variance norm),
    matching your latest CartPole style.
    """
    G = 0.0
    out: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    t = torch.tensor(out, dtype=torch.float32, device=device)
    return t - t.mean()


def run_batch_episode(env: BatchedInvertedPendulumEnv,
                      policy: GaussianPolicyMean,
                      device: torch.device,
                      fixed_std: float,
                      gamma: float,
                      entropy_coef: float = 0.0):
    """
    Roll out ONE episode per environment in parallel until all are done.
    Returns:
      loss (scalar tensor),
      mean_return (float),
      steps_taken (int)
    """
    obs = env.reset()                        # [B,2]
    B = obs.shape[0]
    done_prev = np.zeros((B,), dtype=bool)

    # per-env trajectories
    rewards_per_env: List[List[float]] = [[] for _ in range(B)]
    logps_per_env: List[List[torch.Tensor]] = [[] for _ in range(B)]
    ents_per_env: List[List[torch.Tensor]] = [[] for _ in range(B)]

    std_t = torch.tensor(fixed_std, dtype=torch.float32, device=device)
    steps = 0

    while True:
        # stop when all envs are done (the env returns done=True after termination)
        if done_prev.all():
            break

        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)  # [B,2]
        mean = policy(obs_t).squeeze(-1)  # [B]
        dist = torch.distributions.Normal(loc=mean, scale=std_t)
        action = dist.sample()                           # [B]
        log_prob = dist.log_prob(action)                 # [B]
        entropy = dist.entropy()                         # [B]

        a_np = action.detach().cpu().numpy().astype(np.float32)  # [B]
        next_obs, r, done, _ = env.step(a_np)  # r:[B], done:[B]

        # append to trajectories only for envs that were not done before this step
        active = ~done_prev
        if np.any(active):
            act_idx = np.where(active)[0]
            for i in act_idx:
                rewards_per_env[i].append(float(r[i]))
                logps_per_env[i].append(log_prob[i])
                ents_per_env[i].append(entropy[i])

        obs = next_obs
        done_prev = done
        steps += 1

    # compute per-env losses, then average across the batch
    losses = []
    returns = []
    for i in range(B):
        G = compute_returns(rewards_per_env[i], gamma=gamma, device=device)  # [Ti]
        lp = torch.stack(logps_per_env[i])                                   # [Ti]
        loss_i = -(lp * G).sum()
        if entropy_coef and entropy_coef > 0.0:
            ent = torch.stack(ents_per_env[i]).sum()
            loss_i = loss_i - entropy_coef * ent
        losses.append(loss_i)
        returns.append(sum(rewards_per_env[i]))

    loss = torch.stack(losses).mean()                 # average across batch
    mean_return = float(np.mean(returns))             # mean return across batch

    return loss, mean_return, steps


def train():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env & policy
    env = BatchedInvertedPendulumEnv(batch_size=cfg.batch_size, seed=cfg.seed, config=InvertedPendulumConfig())
    obs_dim = env.observation_space_shape[0]
    policy = GaussianPolicyMean(obs_dim, hidden=cfg.hidden_size).to(device)
    optimizer = optim.SGD(policy.parameters(), lr=cfg.lr)

    # Training loop
    means = []
    for ep in range(1, cfg.episodes + 1):
        loss, mean_ret, steps = run_batch_episode(
            env, policy, device,
            fixed_std=cfg.fixed_std,
            gamma=cgf.gamma if (cgf := cfg) else 0.99,  # keep line short, use cfg.gamma
            entropy_coef=cfg.entropy_coef
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        means.append(mean_ret)
        if cfg.print_every and (ep % cfg.print_every == 0 or ep == 1):
            print(f"Episode {ep:4d} | steps≈{steps:3d} | mean_return(B={cfg.batch_size})={mean_ret:6.1f} | loss={loss.item():.3f}")

    # Plot learning curve (mean return across batch)
    xs = np.arange(1, cfg.episodes + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(xs, means, label=f"Mean return (batch={cfg.batch_size})")
    plt.xlabel("Episode")
    plt.ylabel("Mean return across batch")
    plt.title("Batched REINFORCE on Inverted Pendulum")
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve_batched_ip.png", dpi=150)
    print("Saved plot to learning_curve_batched_ip.png")


if __name__ == "__main__":
    train()
