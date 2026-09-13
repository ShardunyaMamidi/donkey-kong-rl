import gymnasium as gym
import ale_py
import numpy as np
from typing import NamedTuple

from wrappers import FrameSkip, ResizeObservation, FrameStack, ClipReward
from model import DQN
from agent import EpsilonGreedy
from replay_buffer import ReplayBuffer

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

    net = DQN(num_actions=wrapped_env.action_space.n, num_frames=4)
    agent = EpsilonGreedy(epsilon=0.1, rng=seed)

    action = agent.choose_action(wrapped_env.action_space, stacked, net)
    print("chosen action:", action)

    stacked, reward, terminated, truncated, info = wrapped_env.step(action)
    print("after one step:", stacked.shape, stacked.dtype, "reward:", reward)

    replay_buffer = ReplayBuffer(capacity=1000)

    state = stacked
    for _ in range(40):
        action = agent.choose_action(wrapped_env.action_space, state, net)
        next_state, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, info = wrapped_env.reset()

    print("buffer size:", len(replay_buffer))

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size=32)
    print("states:", states.shape, states.dtype)
    print("actions:", actions.shape, actions.dtype)
    print("rewards:", rewards.shape, rewards.dtype)
    print("next_states:", next_states.shape, next_states.dtype)
    print("dones:", dones.shape, dones.dtype)
