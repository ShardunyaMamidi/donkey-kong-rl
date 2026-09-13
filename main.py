from typing import NamedTuple


class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    seed: int
    batch_size: int
    buffer_capacity: int
    warmup_steps: int
    target_sync_steps: int
    num_frames: int
    frame_skip: int


# modest values for a single-game hobby run, not the original DQN paper's
# multi-million-frame scale
params = Params(
    total_episodes=500,
    learning_rate=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=50_000,
    seed=123,
    batch_size=32,
    buffer_capacity=20_000,
    warmup_steps=1_000,
    target_sync_steps=1_000,
    num_frames=4,
    frame_skip=4,
)

if __name__ == "__main__":
    # deferred import: train.py imports Params from this module, so importing
    # it at the top of main.py would create a circular import
    from train import train

    train(params)
