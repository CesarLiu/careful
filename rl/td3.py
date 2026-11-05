"""TD3 (Fujimoto et al., 2018) for the SDC policy, as CAT trains it
(saferl_algo/TD3.py: actor and twin critics with two hidden layers of 256
ReLU units, Adam at 3e-4, discount 0.99, target rate 0.005, target-policy
noise 0.2 clipped at 0.5, delayed actor updates every 2 critic updates),
with the replay buffer on the device and transitions added a batch at a
time, since the simulator (rl/sim.py) steps many worlds together.

The update rule is saferl_algo's line for line; only the data path differs.
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TD3Settings:
    hidden: int = 256
    lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    max_action: float = 1.0


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden: int = 256):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(state_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(state_dim + action_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1)
        self.l4, self.l5, self.l6 = nn.Linear(state_dim + action_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))


class ReplayBuffer:
    """A ring buffer of (state, action, next_state, reward, not_done) on the device."""

    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6), device: str = "cpu"):
        self.max_size, self.ptr, self.size, self.device = int(max_size), 0, 0, device
        z = lambda d: torch.zeros(self.max_size, d, device=device)  # noqa: E731
        self.state, self.action, self.next_state = z(state_dim), z(action_dim), z(state_dim)
        self.reward, self.not_done = z(1), z(1)

    def add(self, state, action, next_state, reward, done) -> torch.Tensor:
        """Adds a batch [n, ...]; returns the n buffer indices written."""
        n = state.shape[0]
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.max_size
        self.state[idx], self.action[idx], self.next_state[idx] = state, action, next_state
        self.reward[idx] = reward.reshape(-1, 1).float()
        self.not_done[idx] = 1.0 - done.reshape(-1, 1).float()
        self.ptr = int((self.ptr + n) % self.max_size)
        self.size = min(self.size + n, self.max_size)
        return idx

    def sample(self, batch_size: int):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.state[ind], self.action[ind], self.next_state[ind], self.reward[ind], self.not_done[ind]


class TD3:
    def __init__(self, state_dim: int, action_dim: int, settings: TD3Settings = TD3Settings(), device: str = "cpu"):
        s = self.s = settings
        self.device = device
        self.actor = Actor(state_dim, action_dim, s.max_action, s.hidden).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=s.lr)
        self.critic = Critic(state_dim, action_dim, s.hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=s.lr)
        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> None:
        s = self.s
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * s.policy_noise).clamp(-s.noise_clip, s.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-s.max_action, s.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + not_done * s.discount * torch.min(target_q1, target_q2)
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.total_it % s.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for net, target in ((self.critic, self.critic_target), (self.actor, self.actor_target)):
                for param, target_param in zip(net.parameters(), target.parameters()):
                    target_param.data.copy_(s.tau * param.data + (1 - s.tau) * target_param.data)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(), "total_it": self.total_it}, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.total_it = state["total_it"]
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
