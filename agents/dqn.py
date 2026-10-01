import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from agents.base import BaseAgent
from agents.exploration import EpsilonGreedy, linear_epsilon
from buffers.replay_buffer import ReplayBuffer
from config import BaseConfig
from networks.q_network import QNetwork


# modest values for a single-game hobby run, not the original DQN paper's
# multi-million-frame scale
@dataclass(frozen=True, kw_only=True)
class DQNConfig(BaseConfig):
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 50_000
    batch_size: int = 32
    buffer_capacity: int = 20_000
    warmup_steps: int = 1_000
    target_sync_steps: int = 1_000


# Copies the original network into a duplicate
def make_target_network(net):
    target_net = copy.deepcopy(net)
    target_net.eval()
    for param in target_net.parameters():
        param.requires_grad_(False)
    return target_net


class DQNAgent(BaseAgent):
    name = "dqn"
    config_cls = DQNConfig

    def __init__(self, cfg, observation_space, action_space, device):
        super().__init__(cfg, observation_space, action_space, device)
        self.net = QNetwork(
            num_actions=action_space.n, num_frames=cfg.num_frames
        ).to(device)
        self.target_net = make_target_network(self.net)
        self.policy = EpsilonGreedy(
            epsilon=cfg.epsilon_start,
            rng=np.random.default_rng(cfg.seed),
            device=device,
        )
        self.buffer = ReplayBuffer(capacity=cfg.buffer_capacity)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.env_steps = 0
        self.train_steps = 0

    def act(self, obs, explore=True):
        if not explore:
            return self.policy.greedy_action(obs, self.net)

        # epsilon value updated based on the no of steps
        self.policy.epsilon = linear_epsilon(
            self.env_steps, self.cfg.epsilon_start, self.cfg.epsilon_end,
            self.cfg.epsilon_decay_steps,
        )
        return self.policy.choose_action(self.action_space, obs, self.net)

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        self.buffer.push(obs, action, reward, next_obs, terminated or truncated)
        self.env_steps += 1

    def update(self):
        # the purpose of warmup_steps is to fill the buffer with states inorder to start learning
        if len(self.buffer) < self.cfg.warmup_steps:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.cfg.batch_size
        )

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        # Q(s, a) for the actions actually taken
        predicted_q = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.cfg.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(predicted_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        # for every target_sync_steps, we copy the weights of original into target
        if self.train_steps % self.cfg.target_sync_steps == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    # target_net is rebuilt from net on load, so only net needs saving
    def checkpoint_modules(self):
        return {"net": self.net}

    def log_stats(self):
        return {"epsilon": self.policy.epsilon}
