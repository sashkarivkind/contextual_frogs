#!/usr/bin/env python3
"""
Vanilla Policy Gradient (REINFORCE) on an in-house CartPole with:
  - Tweak 1: Optional prev-action augmentation (append last action one-hot to obs)
  - Tweak 2: Environment parameter switching at user-defined episodes (ABAB, etc.)
  - Multi-seed experiments for (prev-action ON/OFF), learning-curve plots and CSV.

Usage:
    python reinforce_cartpole_abab.py

Outputs:
  - results_ABAB.csv  : per-episode returns for each seed and condition
  - learning_curves_ABAB.png : mean ± std returns across seeds for both conditions
  - policy_*          : example saved policies from last run (optional, off by default)

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
# In-house CartPole Environment
# =========================================

@dataclass
class CartPoleConfig:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5  # actually half the pole's length
    force_mag: float = 10.0
    tau: float = 0.02  # seconds between state updates

    # Termination thresholds (classic CartPole values)
    theta_threshold_radians: float = 12 * (2 * math.pi / 360.0)  # 12 degrees
    x_threshold: float = 2.4

    # Episode length cap (like Gym v1 uses 500)
    max_steps: int = 500

    # Initial state noise
    init_state_low: float = -0.05
    init_state_high: float = 0.05


class CartPoleEnv:
    """
    Minimal CartPole implementation closely following OpenAI Gym's classic control.
    State: [x, x_dot, theta, theta_dot]
    Actions: 0 (push left), 1 (push right)
    Reward: +1 per step until termination.
    Termination: pole angle or cart position beyond thresholds, or step cap.
    """
    def __init__(self, seed: int | None = None, config: CartPoleConfig | None = None):
        self.cfg = config if config is not None else CartPoleConfig()
        self._rng_seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.np_random = np.random.RandomState(self._rng_seed)
        self.state: np.ndarray | None = None
        self.steps: int = 0

        self.total_mass = self.cfg.masspole + self.cfg.masscart
        self.polemass_length = self.cfg.masspole * self.cfg.length

    @property
    def action_space_n(self) -> int:
        return 2

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        return (4,)

    def set_config(self, new_cfg: CartPoleConfig):
        """Switch environment physics/config on the fly."""
        self.cfg = new_cfg
        self.total_mass = self.cfg.masspole + self.cfg.masscart
        self.polemass_length = self.cfg.masspole * self.cfg.length
        # Note: state remains; in this framework we switch at episode boundaries.

    def reset(self) -> np.ndarray:
        low, high = self.cfg.init_state_low, self.cfg.init_state_high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(np.float32)
        self.steps = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.state is not None, "Call reset() before step()."
        assert action in (0, 1), "Invalid action. Must be 0 (left) or 1 (right)."

        x, x_dot, theta, theta_dot = self.state.tolist()

        force = self.cfg.force_mag if action == 1 else -self.cfg.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Equations of motion (from Gym's cartpole)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.cfg.gravity * sintheta - costheta * temp) / (
            self.cfg.length * (4.0 / 3.0 - self.cfg.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.cfg.tau * x_dot
        x_dot = x_dot + self.cfg.tau * xacc
        theta = theta + self.cfg.tau * theta_dot
        theta_dot = theta_dot + self.cfg.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        done = self._is_done(x, theta) or (self.steps >= self.cfg.max_steps)
        reward = 1.0

        return self._get_obs(), reward, done, {}

    def _is_done(self, x: float, theta: float) -> bool:
        if x < -self.cfg.x_threshold or x > self.cfg.x_threshold:
            return True
        if theta < -self.cfg.theta_threshold_radians or theta > self.cfg.theta_threshold_radians:
            return True
        return False

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        return self.state.copy()


# =========================================
# Tweak 1: Prev-Action Augmentation Wrapper
# =========================================

class PrevActionAugmentedEnv:
    """
    Wraps an env with action_space_n=2. Observation becomes [obs(4), prev_action_onehot(2)].
    On reset: prev_action = [0,0]. After each step, prev_action is set to the action just taken.
    If use_prev_action=False, prev_action is always [0,0] (ablation).
    """
    def __init__(self, 
                 env: CartPoleEnv,
                 use_prev_action: bool = False):
        self.env = env
        self.prev_action_oh = np.zeros(2, dtype=np.float32)
        self.use_prev_action = use_prev_action

    @property
    def action_space_n(self) -> int:
        return self.env.action_space_n

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        base = self.env.observation_space_shape[0]
        return (base + 2,)

    def set_config(self, new_cfg: CartPoleConfig):
        self.env.set_config(new_cfg)

    def reset(self) -> np.ndarray:
        self.prev_action_oh[:] = 0.0
        base_obs = self.env.reset()
        return np.concatenate([base_obs, self.prev_action_oh], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, r, done, info = self.env.step(action)
        # Update prev action AFTER step, so next obs includes the action just taken
        self.prev_action_oh[:] = 0.0
        self.prev_action_oh[action] = 1.0 if self.use_prev_action else 0.0
        aug_obs = np.concatenate([obs, self.prev_action_oh], dtype=np.float32)
        return aug_obs, r, done, info


# =========================================
# Policy (PyTorch)
# =========================================

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


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
    # ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
    ret_t = (ret_t - ret_t.mean())
    return ret_t


def run_episode(env, policy: PolicyNetwork, device: torch.device):
    obs = env.reset()
    done = False
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    ep_len = 0

    while not done:
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)
        logits = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        obs, reward, done, _ = env.step(action.item())
        log_probs.append(log_prob.squeeze(0))
        rewards.append(float(reward))
        ep_len += 1

    return log_probs, rewards, ep_len


# =========================================
# Tweak 2: Phase Schedules (ABAB, etc.)
# =========================================

@dataclass
class PhaseSchedule:
    """
    Define phases as a list of tuples: [('A', len_A), ('B', len_B), ...]
    The total #episodes is the sum of lengths. phase_for_episode(1-based) -> 'A'/'B'/...
    """
    phases: List[tuple[str, int]]

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
        """Episode indices (1-based) where a new phase starts (including ep 1)."""
        pts = []
        cursor = 0
        for _, length in self.phases:
            pts.append(cursor + 1)
            cursor += length
        return pts


def make_ABAB_schedule(len_A: int, len_B: int, cycles: int = 2) -> PhaseSchedule:
    phases = []
    for c in range(cycles):
        phases.append(('A', len_A))
        phases.append(('B', len_B))
    return PhaseSchedule(phases=phases)


# =========================================
# Single Run (one seed, one condition)
# =========================================

@dataclass
class TrainConfig:
    gamma: float = 0.99
    lr: float = 1e-2
    hidden_size: int = 128
    grad_clip: float = 1.0
    print_every: int = 0  # 0 = silent


def single_run(seed: int,
               use_prev_action: bool,
               base_cfg_A: CartPoleConfig,
               base_cfg_B: CartPoleConfig,
               schedule: PhaseSchedule,
               train_cfg: TrainConfig,
               device: torch.device) -> Dict[str, Any]:
    """
    Train one policy across a schedule of env switches. Returns per-episode returns.
    """
    set_seed(seed)

    # Env
    core_env = CartPoleEnv(seed=seed, config=base_cfg_A)
    env = PrevActionAugmentedEnv(core_env, use_prev_action=use_prev_action) 

    # Policy
    # obs_dim = env.observation_space_shape[0] if hasattr(env, "observation_space_shape") else 4 + (2 if use_prev_action else 0)
    obs_dim = env.observation_space_shape[0] if hasattr(env, "observation_space_shape") else 4 + (2 if use_prev_action else 0)
    action_dim = 2
    policy = PolicyNetwork(obs_dim, hidden=train_cfg.hidden_size, action_dim=action_dim).to(device)
    optimizer = optim.SGD(policy.parameters(), lr=train_cfg.lr)

    total_eps = schedule.total_episodes()
    returns_history: List[float] = []
    phases_per_ep: List[str] = []

    for ep in range(1, total_eps + 1):
        phase = schedule.phase_for_episode(ep)
        phases_per_ep.append(phase)
        # Switch config at episode boundary
        if phase == 'A':
            env.set_config(base_cfg_A) if hasattr(env, "set_config") else core_env.set_config(base_cfg_A)
        else:
            env.set_config(base_cfg_B) if hasattr(env, "set_config") else core_env.set_config(base_cfg_B)

        # Collect one episode
        log_probs, rewards, ep_len = run_episode(env, policy, device)
        ep_return = float(sum(rewards))
        returns_history.append(ep_return)

        # REINFORCE update
        G = compute_returns(rewards, gamma=train_cfg.gamma, device=device)  # [T]
        log_probs_t = torch.stack(log_probs)  # [T]
        loss = -(log_probs_t * G).sum()

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
        # "policy": policy.state_dict(),  # uncomment if you want to keep the final weights in memory
    }


# =========================================
# Experiment: Multi-seed, two conditions
# =========================================

def run_experiment(
    seeds: Sequence[int],
    use_prev_action_flags: Sequence[bool],
    schedule: PhaseSchedule,
    cfg_A: CartPoleConfig,
    cfg_B: CartPoleConfig,
    train_cfg: TrainConfig,
    save_policies: bool = False,
    out_csv: str = "results_ABAB.csv",
    out_png: str = "learning_curves_ABAB.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_runs: List[Dict[str, Any]] = []

    for use_prev_action in use_prev_action_flags:
        for seed in seeds:
            res = single_run(seed, use_prev_action, cfg_A, cfg_B, schedule, train_cfg, device)
            all_runs.append(res)
            if save_policies:
                # If desired, save final policy per run (note: we didn't return it above)
                pass

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
    labels = {0: "No prev-action", 1: "Prev-action (2D) appended"}

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
    # Annotate phase bands
    cursor = 0
    for label, length in schedule.phases:
        mid = cursor + length / 2
        plt.text(mid, plt.ylim()[1] * 0.97, label, ha="center", va="top")
        cursor += length

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curves under ABAB Environment Switching (mean ± std across seeds)")
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
    cycles = 2  # ABAB
    schedule = make_ABAB_schedule(len_A=len_A, len_B=len_B, cycles=cycles)

    # ----- Environment configs A & B -----
    # A = classic-ish
    cfg_A = CartPoleConfig(
        gravity=9.8,
        masscart=1.0,
        masspole=0.1,
        length=0.5,
        force_mag=10.0,
        tau=0.02,
        theta_threshold_radians=12 * (2 * math.pi / 360.0),
        x_threshold=2.4,
        max_steps=300,
        init_state_low=-0.05,
        init_state_high=0.05,
    )
    # B = "harder" (different gravity/force/mass) to force re-learning
    cfg_B = replace(cfg_A, masspole=0.2, force_mag=8.0)

    # ----- Training hyperparams -----
    train_cfg = TrainConfig(
        gamma=0.99,
        lr=1e-1,
        hidden_size=128,
        grad_clip=1.0,
        print_every=20,  # set >0 to see live logs
    )

    # ----- Experiment settings -----
    seeds = [0, 1, 2]  # a few seeds
    use_prev_action_flags = [False, True]  # compare tweak1 OFF vs ON

    run_experiment(
        seeds=seeds,
        use_prev_action_flags=use_prev_action_flags,
        schedule=schedule,
        cfg_A=cfg_A,
        cfg_B=cfg_B,
        train_cfg=train_cfg,
        save_policies=False,
        out_csv="results_ABAB.csv",
        out_png="learning_curves_ABAB.png",
    )


if __name__ == "__main__":
    main()
