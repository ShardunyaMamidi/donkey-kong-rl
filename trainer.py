import os

import torch

from envs import make_env
from factory import make_agent, make_config


# algorithm-agnostic training loop: env stepping, episode bookkeeping, logging
# and checkpointing live here; everything algorithm-specific goes through the
# BaseAgent hooks (act / observe / update)
def train(algo, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(cfg)
    agent = make_agent(algo, cfg, env, device)

    global_step = 0
    for episode in range(cfg.total_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.observe(state, action, reward, next_state, terminated, truncated)
            agent.update()

            state = next_state
            episode_reward += reward
            global_step += 1

        stats = "".join(f"{key}={value:.3f}, " for key, value in agent.log_stats().items())
        print(f"episode {episode}: reward={episode_reward}, {stats}steps={global_step}")

        if (episode + 1) % cfg.checkpoint_every == 0:
            path = os.path.join(cfg.checkpoint_dir, f"{algo}_episode_{episode + 1}.pt")
            torch.save(agent.checkpoint(), path)
            print(f"saved checkpoint: {path}")

    final_path = os.path.join(cfg.checkpoint_dir, f"{algo}_final.pt")
    torch.save(agent.checkpoint(), final_path)
    print(f"saved final checkpoint: {final_path}")
    env.close()


if __name__ == "__main__":
    # small values here so this runs quickly as a smoke test; use main.py for
    # a real training run with the config's default values
    smoke_cfg = make_config(
        "dqn",
        total_episodes=2,
        epsilon_decay_steps=100,
        batch_size=8,
        buffer_capacity=200,
        warmup_steps=20,
        checkpoint_dir="checkpoints/smoke",
    )
    train("dqn", smoke_cfg)
