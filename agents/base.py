from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import ClassVar

from config import BaseConfig


# Interface every algorithm implements so the shared trainer (trainer.py) can
# drive it without knowing which algorithm it is. Per env step the trainer calls:
#
#     action = agent.act(obs)
#     agent.observe(obs, action, reward, next_obs, terminated, truncated)
#     agent.update()
#
# Off-policy agents (DQN) store the transition in a replay buffer and take a
# gradient step inside update() every step once warmed up. On-policy agents
# (PPO) can cache log-prob/value from act(), append to a rollout in observe(),
# and make update() a no-op until the rollout is full.
class BaseAgent(ABC):
    name: ClassVar[str]
    config_cls: ClassVar[type[BaseConfig]]

    def __init__(self, cfg, observation_space, action_space, device):
        self.cfg = cfg
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    # explore=False -> pure exploitation, used for playback/evaluation
    @abstractmethod
    def act(self, obs, explore=True) -> int: ...

    @abstractmethod
    def observe(self, obs, action, reward, next_obs, terminated, truncated) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    # name -> module for every network that needs saving to replay the agent
    @abstractmethod
    def checkpoint_modules(self) -> dict: ...

    # extra values the trainer prints at the end of each episode
    def log_stats(self) -> dict:
        return {}

    # algo name + config go in alongside the weights, so play.py can rebuild
    # the right agent from a checkpoint file alone
    def checkpoint(self):
        return {
            "algo": self.name,
            "config": asdict(self.cfg),
            "modules": {
                key: module.state_dict()
                for key, module in self.checkpoint_modules().items()
            },
        }

    def load_checkpoint(self, checkpoint):
        if checkpoint["algo"] != self.name:
            raise ValueError(
                f"checkpoint is for {checkpoint['algo']!r}, not {self.name!r}"
            )
        for key, module in self.checkpoint_modules().items():
            module.load_state_dict(checkpoint["modules"][key])
