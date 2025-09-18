#!/usr/bin/env python3
"""
Vanilla Policy Gradient (REINFORCE) on a fully in-house CartPole environment.
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


# ================================
# In-house CartPole Environment
# ================================

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
        self.np_random = np.random.RandomState(seed if seed is not None else np.random.randint(0, 2**31 - 1))
        self.state: np.ndarray | None = None
        self.steps: int = 0

        # Precompute reusable terms
        self.total_mass = self.cfg.masspole + self.cfg.masscart
        self.polemass_length = self.cfg.masspole * self.cfg.length

    @property
    def action_space_n(self) -> int:
        return 2

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        return (4,)

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

        # Euler update
        x = x + self.cfg.tau * x_dot
        x_dot = x_dot + self.cfg.tau * xacc
        theta = theta + self.cfg.tau * theta_dot
        theta_dot = theta_dot + self.cfg.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        done = self._is_done(x, theta) or (self.steps >= self.cfg.max_steps)
        reward = 1.0  # classic CartPole reward

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

    # Optional: simple text render for debugging (commented out by default)
    # def render(self):
    #     x, x_dot, theta, theta_dot = self.state
    #     print(f"x={x:+.2f} xd={x_dot:+.2f} th={theta:+.2f} thd={theta_dot:+.2f} step={self.steps}")


# ================================
# Policy (PyTorch)
# ================================

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

        # Better initializations can help stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits


# ================================
# Training Utilities
# ================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(policy: PolicyNetwork, device: torch.device, episodes: int = 10, seed: int | None = None) -> float:
    """Average return over a few eval episodes with deterministic action selection (argmax)."""
    env = CartPoleEnv(seed=seed)
    policy.eval()
    total = 0.0
    for _ in range(episodes):
        obs = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            action = torch.argmax(logits, dim=-1).item()
            obs, r, done, _ = env.step(action)
            ep_ret += r
    #         env.render()
        total += ep_ret
    policy.train()
    return total / episodes


def compute_returns(rewards: List[float], gamma: float, device: torch.device) -> torch.Tensor:
    """Compute reward-to-go (returns) for REINFORCE."""
    G = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    # Normalize for variance reduction (still "vanilla" PG; not a learned baseline)
    ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
    return ret_t


def run_episode(env: CartPoleEnv, policy: PolicyNetwork, device: torch.device):
    """Collect one episode: states, actions, log_probs, rewards."""
    obs = env.reset()
    done = False

    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    ep_len = 0

    while not done:
        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
        logits = policy(obs_t)  # [1, action_dim]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()  # [1]
        log_prob = dist.log_prob(action)  # [1]

        obs, reward, done, _ = env.step(action.item())

        log_probs.append(log_prob.squeeze(0))
        rewards.append(reward)
        ep_len += 1

    return log_probs, rewards, ep_len


# ================================
# Main Training Loop (REINFORCE)
# ================================

def main():
    # --- Hyperparameters ---
    seed = 42
    gamma = 0.99
    lr = 1e-1
    episodes = 1000
    hidden_size = 128
    grad_clip = 1.0
    print_every = 10
    solve_threshold = 475.0  # average over last 10 episodes

    # Setup
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment + Policy + Optimizer
    env = CartPoleEnv(seed=seed)
    obs_dim = env.observation_space_shape[0]
    action_dim = env.action_space_n

    policy = PolicyNetwork(obs_dim, hidden=hidden_size, action_dim=action_dim).to(device)
    optimizer = optim.SGD(policy.parameters(), lr=lr)

    # Training
    returns_history: List[float] = []
    best_eval = -float("inf")
    solved_at = None

    for ep in range(1, episodes + 1):
        # Collect an episode
        log_probs, rewards, ep_len = run_episode(env, policy, device)
        ep_return = float(sum(rewards))
        returns_history.append(ep_return)

        # Compute REINFORCE loss: -E[ log pi(a|s) * G_t ]
        G = compute_returns(rewards, gamma=gamma, device=device)  # [T]
        log_probs_t = torch.stack(log_probs)  # [T]
        loss = -(log_probs_t * G).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        # Logging
        if ep % print_every == 0 or ep == 1:
            last_k = min(10, len(returns_history))
            avg_last_k = sum(returns_history[-last_k:]) / last_k
            print(f"Episode {ep:4d} | len={ep_len:3d} | return={ep_return:6.1f} | "
                  f"avg_last_{last_k}={avg_last_k:6.1f} | loss={loss.item():.3f}")

            # Quick evaluation with greedy policy
            eval_score = evaluate(policy, device=device, episodes=5, seed=seed + 999)
            best_eval = max(best_eval, eval_score)
            print(f"  Eval (greedy, 5 eps): {eval_score:.1f} | Best: {best_eval:.1f}")

        # Check solve condition
        if len(returns_history) >= 10:
            moving_avg = sum(returns_history[-10:]) / 10.0
            if moving_avg >= solve_threshold and solved_at is None:
                solved_at = ep

    # Save model
    torch.save({"model_state_dict": policy.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "hidden_size": hidden_size}, "policy_cartpole.pt")
    print("\nTraining finished.")
    if solved_at is not None:
        print(f"Solved at episode {solved_at} (10-ep moving avg ≥ {solve_threshold}).")
    else:
        print("Did not reach solve threshold within training budget.")
    print("Saved policy to policy_cartpole.pt")

    # Demo a few greedy episodes
    demo_env = CartPoleEnv(seed=seed + 2025)
    demo_eps = 3
    print(f"\nGreedy demo for {demo_eps} episodes:")
    for i in range(demo_eps):
        obs = demo_env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)
                logits = policy(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())
            obs, r, done, _ = demo_env.step(action)
            ep_ret += r
            # demo_env.render()  # uncomment to print state
        print(f"  Demo episode {i+1}: return={ep_ret:.1f}")


if __name__ == "__main__":
    main()
