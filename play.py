import argparse

import torch

from envs import make_env
from factory import make_agent, make_config


def play(checkpoint_path, num_episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # the checkpoint records its algorithm + config, so it alone is enough to
    # rebuild a matching env and agent
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = make_config(checkpoint["algo"], **checkpoint["config"])

    env = make_env(cfg, render_mode="human")
    agent = make_agent(checkpoint["algo"], cfg, env, device)
    agent.load_checkpoint(checkpoint)

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # explore=False -> always exploit the trained network, no random exploration
            action = agent.act(state, explore=False)
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
