from dataclasses import dataclass


# settings every algorithm shares (run length, env preprocessing, checkpointing);
# each algorithm subclasses this with its own hyperparameters, e.g. DQNConfig
# in agents/dqn.py. kw_only lets subclasses add fields without default-order
# headaches.
@dataclass(frozen=True, kw_only=True)
class BaseConfig:
    total_episodes: int = 500
    seed: int = 123
    num_frames: int = 4
    frame_skip: int = 4
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 50
