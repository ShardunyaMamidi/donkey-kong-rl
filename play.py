import argparse

import gymnasium as gym
import ale_py
import numpy as np
import torch

from wrappers import FrameSkip, ResizeObservation, FrameStack, ClipReward
from model import DQN
from agent import EpsilonGreedy
from main import params


def make_render_env(params):
    env = gym.make("ALE/DonkeyKong-v5", obs_type="grayscale", render_mode="human")
    env = FrameSkip(env, skip=params.frame_skip)
    env = ClipReward(env)
    env = ResizeObservation(env, size=(84, 84))
    env = FrameStack(env, num_frames=params.num_frames)
    return env


def play(checkpoint_path, num_episodes=5):
    gym.register_envs(ale_py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_render_env(params)
    net = DQN(num_actions=env.action_space.n, num_frames=params.num_frames).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()

    # epsilon=0 -> always exploit the trained network, no random exploration
    agent = EpsilonGreedy(epsilon=0.0, rng=np.random.default_rng(), device=device)

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.choose_action(env.action_space, state, net)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        print(f"episode {episode}: reward={episode_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dqn_final.pt")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    play(args.checkpoint, args.episodes)
