#!/usr/bin/env python3
"""
Batched REINFORCE (continuous action) for an in-house Inverted Pendulum with:
  - Prev-action augmentation (scalar) on/off, but obs dim kept identical across conditions.
  - ABAB phase switching (episode-wise).
  - Multi-seed experiments with mean±std aggregation.
  - Default batch_size=64; mean return is averaged across the batch.

Outputs:
  - results_IP_batch_ABAB.csv
  - learning_curves_IP_batch_ABAB.png
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


# ============================
# Inverted Pendulum (Batched)
# ============================

@dataclass
class InvertedPendulumConfig:
    gravity: float = 9.81
    mass: float = 1.0
    length: float = 1.0
    damping: float = 0.05
    max_torque: float = 2.0
    tau: float = 0.02
    angle_threshold: float = math.radians(24.0)
    max_steps: int = 400

    init_theta_low: float = math.radians(-5.0)
    init_theta_high: float = math.radians(+5.0)
    init_dtheta_low: float = -0.5
    init_dtheta_high: float = +0.5


class BatchedInvertedPendulumEnv:
    """
    State per env: [theta, theta_dot]
    Action per env: scalar torque u in [-max_torque, max_torque]
    Reward: +1 per alive step, else 0 after done
    """
    def __init__(self, batch_size: int = 64, seed: int | None = None, config: InvertedPendulumConfig | None = None):
        self.cfg = config if config is not None else InvertedPendulumConfig()
        self.batch_size = int(batch_size)
        base_seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.np_randoms = [np.random.RandomState(base_seed + 9973 * i) for i in range(self.batch_size)]

        self.state = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.steps = np.zeros((self.batch_size,), dtype=np.int32)
        self.done = np.zeros((self.batch_size,), dtype=bool)
        self._update_derived()

    def _update_derived(self):
        self.I = self.cfg.mass * (self.cfg.length ** 2)

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        return (2,)

    def set_config(self, new_cfg: InvertedPendulumConfig):
        self.cfg = new_cfg
        self._update_derived()

    def reset(self) -> np.ndarray:
        th = np.array([rng.uniform(self.cfg.init_theta_low, self.cfg.init_theta_high) for rng in self.np_randoms],
                      dtype=np.float32)
        dth = np.array([rng.uniform(self.cfg.init_dtheta_low, self.cfg.init_dtheta_high) for rng in self.np_randoms],
                       dtype=np.float32)
        self.state[:, 0] = th
        self.state[:, 1] = dth
        self.steps[:] = 0
        self.done[:] = False
        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32).reshape(-1)
        assert actions.shape[0] == self.batch_size

        rewards = np.zeros((self.batch_size,), dtype=np.float32)
        clipped_actions = np.zeros((self.batch_size,), dtype=np.float32)

        active = ~self.done
        if np.any(active):
            theta = self.state[active, 0].astype(np.float64)
            theta_dot = self.state[active, 1].astype(np.float64)

            u = np.clip(actions[active], -self.cfg.max_torque, self.cfg.max_torque).astype(np.float64)
            clipped_actions[active] = u.astype(np.float32)

            theta_ddot = (self.cfg.mass * self.cfg.gravity * self.cfg.length * np.sin(theta)
                          + u - self.cfg.damping * theta_dot) / (self.I + 1e-8)

            theta_dot = theta_dot + self.cfg.tau * theta_ddot
            theta = theta + self.cfg.tau * theta_dot
            theta = (theta + np.pi) % (2 * np.pi) - np.pi  # wrap

            self.state[active, 0] = theta.astype(np.float32)
            self.state[active, 1] = theta_dot.astype(np.float32)

            rewards[active] = 1.0
            self.steps[active] += 1

            term_angle = np.abs(self.state[active, 0]) > self.cfg.angle_threshold
            term_steps = self.steps[active] >= self.cfg.max_steps
            newly_done = np.zeros_like(active)
            newly_done[active] = np.logical_or(term_angle, term_steps)
            self.done = np.logical_or(self.done, newly_done)

        return self._get_obs(), rewards, self.done.copy(), {"clipped_actions": clipped_actions}

    def _get_obs(self) -> np.ndarray:
        return self.state.copy()


# ===============================================
# Prev-Action Augmentation (batched, scalar prev_u)
# ===============================================

class PrevActionAugmentedEnvBatched:
    """
    Observation becomes [theta, theta_dot, prev_u].
    We ALWAYS append prev_u so obs dim is identical across conditions.
    If use_prev_action=False, prev_u is held at 0.
    """
    def __init__(self, env: BatchedInvertedPendulumEnv, use_prev_action: bool = False):
        self.env = env
        self.use_prev_action = bool(use_prev_action)
        self.prev_u = np.zeros((self.env.batch_size,), dtype=np.float32)

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        base = self.env.observation_space_shape[0]
        return (base + 1,)

    def set_config(self, new_cfg: InvertedPendulumConfig):
        self.env.set_config(new_cfg)

    def reset(self) -> np.ndarray:
        self.prev_u[:] = 0.0
        base = self.env.reset().astype(np.float32)
        return np.concatenate([base, self.prev_u[:, None]], axis=1)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, r, done, info = self.env.step(actions)
        if self.use_prev_action:
            cu = info.get("clipped_actions", np.zeros_like(self.prev_u))
            self.prev_u = np.asarray(cu, dtype=np.float32)
        else:
            self.prev_u[:] = 0.0
        aug = np.concatenate([obs.astype(np.float32), self.prev_u[:, None]], axis=1)
        return aug, r, done, info


# ============================
# Policy: mean-only Gaussian
# ============================

class GaussianPolicyMean(nn.Module):
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


# ============================
# Utilities
# ============================

@dataclass
class TrainConfig:
    gamma: float = 0.99
    lr: float = 3e-3
    hidden_size: int = 128
    grad_clip: float = 1.0
    fixed_std: float = 0.5
    entropy_coef: float = 1e-3
    print_every: int = 10

    # batching + schedule
    batch_size: int = 64
    seed: int = 0  # per-run base seed (varied across runs in experiment)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_returns(rewards: List[float], gamma: float, device: torch.device) -> torch.Tensor:
    G = 0.0
    out: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    t = torch.tensor(out, dtype=torch.float32, device=device)
    return t - t.mean()


# ============================
# ABAB schedule
# ============================

@dataclass
class PhaseSchedule:
    phases: List[tuple[str, int]]  # e.g. [('A', 200), ('B', 200), ...]

    def total_episodes(self) -> int:
        return sum(n for _, n in self.phases)

    def phase_for_episode(self, ep_idx: int) -> str:
        assert ep_idx >= 1
        k = 0
        for label, length in self.phases:
            if k < ep_idx <= k + length:
                return label
            k += length
        return self.phases[-1][0]

    def switch_points(self) -> List[int]:
        pts, k = [], 0
        for _, length in self.phases:
            pts.append(k + 1)
            k += length
        return pts


def make_ABAB_schedule(len_A: int, len_B: int, cycles: int = 2) -> PhaseSchedule:
    phases = []
    for _ in range(cycles):
        phases.append(('A', len_A))
        phases.append(('B', len_B))
    return PhaseSchedule(phases=phases)


# ============================
# One batched episode (per run)
# ============================

def run_batch_episode(env: PrevActionAugmentedEnvBatched,
                      policy: GaussianPolicyMean,
                      device: torch.device,
                      fixed_std: float,
                      gamma: float,
                      entropy_coef: float = 0.0):
    obs = env.reset()  # [B,3]
    B = obs.shape[0]

    done_prev = np.zeros((B,), dtype=bool)
    rewards_per_env: List[List[float]] = [[] for _ in range(B)]
    logps_per_env: List[List[torch.Tensor]] = [[] for _ in range(B)]
    ents_per_env: List[List[torch.Tensor]] = [[] for _ in range(B)]

    std_t = torch.tensor(fixed_std, dtype=torch.float32, device=device)

    # step until all envs finished
    while True:
        if done_prev.all():
            break

        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)  # [B,obs]
        mean = policy(obs_t).squeeze(-1)  # [B]
        dist = torch.distributions.Normal(loc=mean, scale=std_t)
        action = dist.sample()                          # [B]
        log_prob = dist.log_prob(action)                # [B]
        entropy = dist.entropy()                        # [B]

        a_np = action.detach().cpu().numpy().astype(np.float32)
        next_obs, r, done, info = env.step(a_np)

        active = ~done_prev
        if np.any(active):
            idx = np.where(active)[0]
            for i in idx:
                rewards_per_env[i].append(float(r[i]))
                logps_per_env[i].append(log_prob[i])
                ents_per_env[i].append(entropy[i])

        obs = next_obs
        done_prev = done

    # per-env REINFORCE losses; mean across batch
    losses = []
    returns = []
    for i in range(B):
        G = compute_returns(rewards_per_env[i], gamma=gamma, device=device)
        lp = torch.stack(logps_per_env[i]) if len(logps_per_env[i]) else torch.tensor(0.0, device=device)
        loss_i = -(lp * G).sum() if G.numel() > 0 else torch.tensor(0.0, device=device)
        if entropy_coef and len(ents_per_env[i]):
            ent = torch.stack(ents_per_env[i]).sum()
            loss_i = loss_i - entropy_coef * ent
        losses.append(loss_i)
        returns.append(sum(rewards_per_env[i]))

    loss = torch.stack(losses).mean()
    mean_return = float(np.mean(returns)) if returns else 0.0
    return loss, mean_return


# ============================
# Single run (one seed, one flag)
# ============================

def single_run(seed: int,
               use_prev_action: bool,
               cfg_A: InvertedPendulumConfig,
               cfg_B: InvertedPendulumConfig,
               schedule: PhaseSchedule,
               train_cfg: TrainConfig,
               device: torch.device) -> Dict[str, Any]:

    set_seed(seed)

    core_env = BatchedInvertedPendulumEnv(batch_size=train_cfg.batch_size, seed=seed, config=cfg_A)
    env = PrevActionAugmentedEnvBatched(core_env, use_prev_action=use_prev_action)

    obs_dim = env.observation_space_shape[0]  # base(2)+1 = 3
    policy = GaussianPolicyMean(obs_dim, hidden=train_cfg.hidden_size).to(device)
    optimizer = optim.SGD(policy.parameters(), lr=train_cfg.lr)

    total_eps = schedule.total_episodes()
    mean_returns: List[float] = []
    phases_per_ep: List[str] = []

    for ep in range(1, total_eps + 1):
        phase = schedule.phase_for_episode(ep)
        phases_per_ep.append(phase)
        env.set_config(cfg_A if phase == 'A' else cfg_B)

        loss, mean_ret = run_batch_episode(
            env, policy, device,
            fixed_std=train_cfg.fixed_std,
            gamma=train_cfg.gamma,
            entropy_coef=train_cfg.entropy_coef
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.grad_clip)
        optimizer.step()

        mean_returns.append(mean_ret)
        if train_cfg.print_every and (ep % train_cfg.print_every == 0 or ep == 1):
            print(f"[seed={seed}][prev_action={use_prev_action}] ep={ep}/{total_eps} "
                  f"phase={phase} mean_return(B={train_cfg.batch_size})={mean_ret:6.1f} loss={loss.item():.3f}")

    return {
        "seed": seed,
        "use_prev_action": use_prev_action,
        "mean_returns": mean_returns,
        "phases": phases_per_ep,
    }


# ============================
# Experiment (seeds × flags)
# ============================

def run_experiment(
    seeds: Sequence[int],
    use_prev_action_flags: Sequence[bool],
    schedule: PhaseSchedule,
    cfg_A: InvertedPendulumConfig,
    cfg_B: InvertedPendulumConfig,
    train_cfg: TrainConfig,
    out_csv: str = "results_IP_batch_ABAB.csv",
    out_png: str = "learning_curves_IP_batch_ABAB.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_runs: List[Dict[str, Any]] = []

    for flag in use_prev_action_flags:
        for s in seeds:
            res = single_run(s, flag, cfg_A, cfg_B, schedule, train_cfg, device)
            all_runs.append(res)

    # Save CSV
    total_eps = schedule.total_episodes()
    switch_pts = schedule.switch_points()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "seed", "use_prev_action", "mean_return", "phase"])
        for run in all_runs:
            seed = run["seed"]
            use_prev = int(run["use_prev_action"])
            phases = run["phases"]
            for ep in range(1, total_eps + 1):
                w.writerow([ep, seed, use_prev, run["mean_returns"][ep - 1], phases[ep - 1]])
    print(f"Saved raw results to {os.path.abspath(out_csv)}")

    # Aggregate mean±std over seeds per condition
    def aggregate(flag: bool):
        runs = [r for r in all_runs if r["use_prev_action"] == flag]
        mat = np.array([r["mean_returns"] for r in runs], dtype=np.float32)  # [num_seeds, episodes]
        return mat.mean(axis=0), mat.std(axis=0)

    means, stds = {}, {}
    labels = {0: "No prev-action", 1: "Prev-action appended (scalar)"}
    for flag in use_prev_action_flags:
        m, s = aggregate(flag)
        means[flag] = m
        stds[flag] = s

    # Plot
    plt.figure(figsize=(10, 6))
    for flag in use_prev_action_flags:
        xs = np.arange(1, len(means[flag]) + 1)
        plt.plot(xs, means[flag], label=labels[int(flag)])
        plt.fill_between(xs, means[flag] - stds[flag], means[flag] + stds[flag], alpha=0.2)

    for sp in switch_pts:
        plt.axvline(sp, linestyle="--", alpha=0.35)
    ymax = plt.ylim()[1]
    cursor = 0
    for label, length in schedule.phases:
        mid = cursor + length / 2
        plt.text(mid, ymax * 0.97, label, ha="center", va="top")
        cursor += length

    plt.xlabel("Episode")
    plt.ylabel("Mean return across batch")
    plt.title(f"Inverted Pendulum (Batched {train_cfg.batch_size}): ABAB mean±std over seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {os.path.abspath(out_png)}")


# ============================
# Example main
# ============================

def main():
    # Schedule
    len_A, len_B, cycles = 600, 600, 2
    schedule = make_ABAB_schedule(len_A, len_B, cycles)

    # Env configs
    cfg_A = InvertedPendulumConfig(
        gravity=9.81, mass=1.0, length=1.0, damping=0.05,
        max_torque=2.0, tau=0.02, angle_threshold=math.radians(24.0), max_steps=400,
        init_theta_low=math.radians(-5.0), init_theta_high=math.radians(+5.0),
        init_dtheta_low=-0.5, init_dtheta_high=+0.5
    )
    # Harder B (heavier + more damping + less torque)
    cfg_B = replace(cfg_A, mass=1.5, damping=0.10, max_torque=1.2)

    # Training config
    train_cfg = TrainConfig(
        gamma=0.99, lr=3e-2, hidden_size=128, grad_clip=1.0,
        fixed_std=0.5, entropy_coef=1e-3, print_every=10,
        batch_size=64, seed=0
    )

    # Experiment
    seeds = [1,2,3]                 # multi-seed
    use_prev_action_flags = [True, False]  # compare augmentation OFF vs ON

    run_experiment(
        seeds=seeds,
        use_prev_action_flags=use_prev_action_flags,
        schedule=schedule,
        cfg_A=cfg_A,
        cfg_B=cfg_B,
        train_cfg=train_cfg,
        out_csv="results_IP_batch_ABAB.csv",
        out_png="learning_curves_IP_batch_ABAB.png",
    )


if __name__ == "__main__":
    main()
