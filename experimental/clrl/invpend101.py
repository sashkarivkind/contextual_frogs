#!/usr/bin/env python3
"""
Vanilla Policy Gradient (REINFORCE) on an in-house Inverted Pendulum with:
  - Continuous action space (torque): Gaussian policy with fixed std, mean is 1-D network output
  - Tweak 1: Optional prev-action augmentation (append last torque scalar to obs)
  - Tweak 2: Environment parameter switching at user-defined episodes (ABAB, etc.)
  - Multi-seed experiments for (prev-action ON/OFF), learning-curve plots and CSV.

Usage:
    python reinforce_inverted_pendulum_abab.py

Outputs:
  - results_IP_ABAB.csv         : per-episode returns for each seed and condition
  - learning_curves_IP_ABAB.png : mean ± std returns across seeds for both conditions

Author: you + ChatGPT
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, replace
from typing import Tuple, Dict, Any, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import os


# =========================================
# In-house Inverted Pendulum Environment
# =========================================

@dataclass
class InvertedPendulumConfig:
    gravity: float = 9.81     # m/s^2
    mass: float = 1.0         # kg
    length: float = 1.0       # m (rod length; COM at L/2 if uniform, here we use point-mass at end)
    damping: float = 0.05     # viscous friction at the joint (N*m*s/rad)
    max_torque: float = 2.0   # action bound |u| <= max_torque
    tau: float = 0.02         # s, integration timestep
    angle_threshold: float = math.radians(24.0)  # terminate if |theta| > threshold
    max_steps: int = 400      # episode cap

    # initial state noise around upright (theta ~ 0)
    init_theta_low: float = math.radians(-5.0)
    init_theta_high: float = math.radians(+5.0)
    init_dtheta_low: float = -0.5
    init_dtheta_high: float = +0.5


class InvertedPendulumEnv:
    """
    Simple torque-controlled inverted pendulum (single joint).
    State: [theta, theta_dot], where theta=0 is upright.
    Action: torque u in [-max_torque, max_torque].
    Dynamics: I*theta_ddot = m*g*L*sin(theta) + u - b*theta_dot,
              with I = m*L^2 (point mass at end).
    Reward: +1 per step until termination (survival reward).
    Termination: |theta| > angle_threshold or step >= max_steps.
    """
    def __init__(self, seed: int | None = None, config: InvertedPendulumConfig | None = None):
        self.cfg = config if config is not None else InvertedPendulumConfig()
        self._rng_seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.np_random = np.random.RandomState(self._rng_seed)
        self.state: np.ndarray | None = None
        self.steps: int = 0
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
        th = self.np_random.uniform(self.cfg.init_theta_low, self.cfg.init_theta_high)
        dth = self.np_random.uniform(self.cfg.init_dtheta_low, self.cfg.init_dtheta_high)
        self.state = np.array([th, dth], dtype=np.float32)
        self.steps = 0
        return self._get_obs()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.state is not None, "Call reset() before step()."

        theta, theta_dot = self.state.tolist()

        # clamp action to env bounds
        u = float(np.clip(action, -self.cfg.max_torque, self.cfg.max_torque))

        # dynamics: I*theta_ddot = m*g*L*sin(theta) + u - b*theta_dot
        theta_ddot = (self.cfg.mass * self.cfg.gravity * self.cfg.length * math.sin(theta)
                      + u - self.cfg.damping * theta_dot) / (self.I + 1e-8)

        # semi-implicit (symplectic) Euler tends to be more stable
        theta_dot = theta_dot + self.cfg.tau * theta_ddot
        theta = theta + self.cfg.tau * theta_dot

        # keep theta within [-pi, pi] for numerical sanity (optional)
        if theta > math.pi:
            theta -= 2 * math.pi
        elif theta < -math.pi:
            theta += 2 * math.pi

        self.state = np.array([theta, theta_dot], dtype=np.float32)
        self.steps += 1

        done = (abs(theta) > self.cfg.angle_threshold) or (self.steps >= self.cfg.max_steps)
        reward = 1.0  # survival reward

        return self._get_obs(), reward, done, {"u": u}

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        return self.state.copy()


# =========================================
# Tweak 1: Prev-Action Augmentation Wrapper (scalar)
# =========================================

class PrevActionAugmentedEnvCont:
    """
    Observation becomes [theta, theta_dot, prev_u].
    We ALWAYS append the prev_u slot so obs_dim is identical across conditions.
    If use_prev_action=True, prev_u:= last (clipped) action; else prev_u:=0.
    """
    def __init__(self, env: InvertedPendulumEnv, use_prev_action: bool = False):
        self.env = env
        self.use_prev_action = bool(use_prev_action)
        self.prev_u = 0.0

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        base = self.env.observation_space_shape[0]
        return (base + 1,)

    def set_config(self, new_cfg: InvertedPendulumConfig):
        self.env.set_config(new_cfg)

    def reset(self) -> np.ndarray:
        self.prev_u = 0.0
        base_obs = self.env.reset().astype(np.float32)
        return np.concatenate([base_obs, np.array([self.prev_u], dtype=np.float32)])

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, r, done, info = self.env.step(action)
        # Update prev action AFTER acting
        if self.use_prev_action:
            self.prev_u = float(info.get("u", 0.0))
        else:
            self.prev_u = 0.0
        aug_obs = np.concatenate([obs.astype(np.float32), np.array([self.prev_u], dtype=np.float32)])
        return aug_obs, r, done, info


# =========================================
# Policy (Gaussian mean, fixed std)
# =========================================

class GaussianPolicyMean(nn.Module):
    """
    Output: 1-D mean for torque; std is fixed externally.
    We keep the mean unbounded; the env clamps sampled actions to its torque limits.
    """
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_head = nn.Linear(hidden, 1)
        self._init()

    def _init(self):
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.kaiming_normal_(self.mean_head.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_head(x)  # shape [B,1]
        return mean


# =========================================
# Training Utilities
# =========================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_returns(rewards: List[float], gamma: float, device: torch.device) -> torch.Tensor:
    G = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    # Match your latest style: center only (no variance norm)
    ret_t = ret_t - ret_t.mean()
    return ret_t


def run_episode(env, policy: GaussianPolicyMean, device: torch.device, fixed_std: float, entropy_coef: float = 0.0):
    obs = env.reset()
    done = False
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    rewards: List[float] = []
    ep_len = 0

    std_t = torch.tensor(fixed_std, dtype=torch.float32, device=device)

    while not done:
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,obs]
        mean = policy(obs_t).squeeze(-1)  # [1]
        dist = torch.distributions.Normal(loc=mean, scale=std_t)
        action_sample = dist.sample()          # [1]
        log_prob = dist.log_prob(action_sample)  # [1]
        entropy = dist.entropy()               # [1]

        # use scalar action (env will clamp internally)
        a = float(action_sample.squeeze(0).item())
        obs, reward, done, _ = env.step(a)

        log_probs.append(log_prob.squeeze(0))
        entropies.append(entropy.squeeze(0))
        rewards.append(float(reward))
        ep_len += 1

    return log_probs, entropies, rewards, ep_len


# =========================================
# Phase Schedules (ABAB, etc.)
# =========================================

@dataclass
class PhaseSchedule:
    phases: List[tuple[str, int]]  # e.g., [('A', 150), ('B', 150), ...]

    def total_episodes(self) -> int:
        return sum(length for _, length in self.phases)

    def phase_for_episode(self, ep_idx: int) -> str:
        assert ep_idx >= 1
        cursor = 0
        for label, length in self.phases:
            if cursor < ep_idx <= cursor + length:
                return label
            cursor += length
        return self.phases[-1][0]

    def switch_points(self) -> List[int]:
        pts = []
        cursor = 0
        for _, length in self.phases:
            pts.append(cursor + 1)
            cursor += length
        return pts


def make_ABAB_schedule(len_A: int, len_B: int, cycles: int = 2) -> PhaseSchedule:
    phases = []
    for _ in range(cycles):
        phases.append(('A', len_A))
        phases.append(('B', len_B))
    return PhaseSchedule(phases=phases)


# =========================================
# Single Run (one seed, one condition)
# =========================================

@dataclass
class TrainConfig:
    gamma: float = 0.99
    lr: float = 3e-3
    hidden_size: int = 128
    grad_clip: float = 1.0
    print_every: int = 0  # 0 = silent
    fixed_std: float = 0.5
    entropy_coef: float = 1e-3  # tiny entropy helps in continuous control


def single_run(seed: int,
               use_prev_action: bool,
               base_cfg_A: InvertedPendulumConfig,
               base_cfg_B: InvertedPendulumConfig,
               schedule: PhaseSchedule,
               train_cfg: TrainConfig,
               device: torch.device) -> Dict[str, Any]:
    """
    Train one policy across a schedule of env switches. Returns per-episode returns.
    """
    set_seed(seed)

    # Env
    core_env = InvertedPendulumEnv(seed=seed, config=base_cfg_A)
    env = PrevActionAugmentedEnvCont(core_env, use_prev_action=use_prev_action)

    # Policy
    obs_dim = env.observation_space_shape[0]
    policy = GaussianPolicyMean(obs_dim, hidden=train_cfg.hidden_size).to(device)
    optimizer = optim.SGD(policy.parameters(), lr=train_cfg.lr)

    total_eps = schedule.total_episodes()
    returns_history: List[float] = []
    phases_per_ep: List[str] = []

    for ep in range(1, total_eps + 1):
        phase = schedule.phase_for_episode(ep)
        phases_per_ep.append(phase)
        # Switch config at episode boundary
        if phase == 'A':
            env.set_config(base_cfg_A)
        else:
            env.set_config(base_cfg_B)

        # Collect one episode
        log_probs, entropies, rewards, ep_len = run_episode(
            env, policy, device, fixed_std=train_cfg.fixed_std, entropy_coef=train_cfg.entropy_coef
        )
        ep_return = float(sum(rewards))
        returns_history.append(ep_return)

        # REINFORCE loss with (optional) entropy bonus
        G = compute_returns(rewards, gamma=train_cfg.gamma, device=device)  # [T]
        log_probs_t = torch.stack(log_probs)   # [T]
        loss = -(log_probs_t * G).sum()
        if train_cfg.entropy_coef and train_cfg.entropy_coef > 0.0:
            entropy_t = torch.stack(entropies).sum()
            loss = loss - train_cfg.entropy_coef * entropy_t

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if train_cfg.print_every and (ep % train_cfg.print_every == 0 or ep == 1):
            print(f"[seed={seed}][prev_action={use_prev_action}] ep={ep}/{total_eps} "
                  f"phase={phase} len={ep_len} return={ep_return:.1f} loss={loss.item():.3f}")

    return {
        "seed": seed,
        "use_prev_action": use_prev_action,
        "returns": returns_history,
        "phases": phases_per_ep,
    }


# =========================================
# Experiment: Multi-seed, two conditions
# =========================================

def run_experiment(
    seeds: Sequence[int],
    use_prev_action_flags: Sequence[bool],
    schedule: PhaseSchedule,
    cfg_A: InvertedPendulumConfig,
    cfg_B: InvertedPendulumConfig,
    train_cfg: TrainConfig,
    out_csv: str = "results_IP_ABAB.csv",
    out_png: str = "learning_curves_IP_ABAB.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_runs: List[Dict[str, Any]] = []

    for use_prev_action in use_prev_action_flags:
        for seed in seeds:
            res = single_run(seed, use_prev_action, cfg_A, cfg_B, schedule, train_cfg, device)
            all_runs.append(res)

    # Save CSV of raw results
    total_eps = schedule.total_episodes()
    switch_pts = schedule.switch_points()
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "seed", "use_prev_action", "return", "phase"])
        for run in all_runs:
            seed = run["seed"]
            use_prev = int(run["use_prev_action"])
            phases = run["phases"]
            for ep in range(1, total_eps + 1):
                writer.writerow([ep, seed, use_prev, run["returns"][ep - 1], phases[ep - 1]])
    print(f"Saved raw results to {os.path.abspath(out_csv)}")

    # Aggregate (mean ± std over seeds) per condition
    def aggregate(condition_flag: bool):
        runs = [r for r in all_runs if r["use_prev_action"] == condition_flag]
        mat = np.array([r["returns"] for r in runs], dtype=np.float32)  # [num_seeds, episodes]
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        return mean, std

    means = {}
    stds = {}
    labels = {0: "No prev-action", 1: "Prev-action appended (scalar)"}

    for flag in [False, True]:
        if flag in use_prev_action_flags:
            m, s = aggregate(flag)
            means[flag] = m
            stds[flag] = s

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    for flag in use_prev_action_flags:
        m = means[flag]
        s = stds[flag]
        xs = np.arange(1, len(m) + 1)
        plt.plot(xs, m, label=labels[int(flag)])
        plt.fill_between(xs, m - s, m + s, alpha=0.2)

    # Vertical lines at phase switches + labels on top
    for sp in switch_pts:
        plt.axvline(sp, linestyle="--", alpha=0.4)
    cursor = 0
    top = plt.ylim()[1]
    for label, length in schedule.phases:
        mid = cursor + length / 2
        plt.text(mid, top * 0.97, label, ha="center", va="top")
        cursor += length

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Inverted Pendulum: ABAB Learning Curves (mean ± std across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {os.path.abspath(out_png)}")


# =========================================
# Example main() with sensible defaults
# =========================================

def main():
    # ----- Schedule: ABAB -----
    len_A = 200
    len_B = 200
    cycles = 2
    schedule = make_ABAB_schedule(len_A=len_A, len_B=len_B, cycles=cycles)

    # ----- Environment configs A & B -----
    # A: nominal
    cfg_A = InvertedPendulumConfig(
        gravity=9.81,
        mass=1.0,
        length=1.0,
        damping=0.05,
        max_torque=2.0,
        tau=0.02,
        angle_threshold=math.radians(24.0),
        max_steps=400,
        init_theta_low=math.radians(-5.0),
        init_theta_high=math.radians(+5.0),
        init_dtheta_low=-0.5,
        init_dtheta_high=+0.5,
    )
    # B: harder (heavier + lower torque + more damping)
    cfg_B = replace(cfg_A, mass=1.5, max_torque=1.2, damping=0.1)

    # ----- Training hyperparams -----
    train_cfg = TrainConfig(
        gamma=0.99,
        lr=1e-2,
        hidden_size=128,
        grad_clip=1.0,
        print_every=20,
        fixed_std=0.5,
        entropy_coef=1e-3,
    )

    # ----- Experiment settings -----
    seeds = [0, 1, 2]
    use_prev_action_flags = [False, True]

    run_experiment(
        seeds=seeds,
        use_prev_action_flags=use_prev_action_flags,
        schedule=schedule,
        cfg_A=cfg_A,
        cfg_B=cfg_B,
        train_cfg=train_cfg,
        out_csv="results_IP_ABAB.csv",
        out_png="learning_curves_IP_ABAB.png",
    )


if __name__ == "__main__":
    main()
