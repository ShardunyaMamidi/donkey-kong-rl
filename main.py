import gymnasium as gym
import ale_py
import numpy as np
from typing import NamedTuple

from wrappers import FrameSkip, ResizeObservation, FrameStack, ClipReward

gym.register_envs(ale_py)
seed = np.random.default_rng(123)

env = gym.make(
    "ALE/DonkeyKong-v5",
    obs_type="grayscale"
)

if __name__ == "__main__":
    wrapped_env = FrameSkip(env, skip=4)
    wrapped_env = ClipReward(wrapped_env)
    wrapped_env = ResizeObservation(wrapped_env, size=(84, 84))
    wrapped_env = FrameStack(wrapped_env, num_frames=4)

    stacked, info = wrapped_env.reset()
    print("after reset:", stacked.shape, stacked.dtype)

    action = wrapped_env.action_space.sample()
    stacked, reward, terminated, truncated, info = wrapped_env.step(action)
    print("after one step:", stacked.shape, stacked.dtype, "reward:", reward)
