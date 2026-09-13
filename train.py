import os

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn

from wrappers import FrameSkip, ResizeObservation, FrameStack, ClipReward
from model import DQN
from agent import EpsilonGreedy
from replay_buffer import ReplayBuffer
from main import Params


def make_env(params):
    env = gym.make("ALE/DonkeyKong-v5", obs_type="grayscale")
    env = FrameSkip(env, skip=params.frame_skip)
    env = ClipReward(env)
    env = ResizeObservation(env, size=(84, 84))
    env = FrameStack(env, num_frames=params.num_frames)
    return env


# linear decay from epsilon_start to epsilon_end over epsilon_decay_steps,
# then held constant at epsilon_end
def linear_epsilon(step, epsilon_start, epsilon_end, epsilon_decay_steps):
    fraction = min(step / epsilon_decay_steps, 1.0)
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


# Copies the original network into a duplicate
def make_target_network(net, num_frames):
    target_net = DQN(num_actions=net.fc[-1].out_features, num_frames=num_frames)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    for param in target_net.parameters():
        param.requires_grad_(False)
    return target_net


def train(params, checkpoint_dir="checkpoints", checkpoint_every=50):
    os.makedirs(checkpoint_dir, exist_ok=True)

    gym.register_envs(ale_py)
    rng = np.random.default_rng(params.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(params)
    net = DQN(num_actions=env.action_space.n, num_frames=params.num_frames).to(device)
    target_net = make_target_network(net, params.num_frames).to(device)
    agent = EpsilonGreedy(epsilon=params.epsilon_start, rng=rng, device=device)
    buffer = ReplayBuffer(capacity=params.buffer_capacity)

    optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)
    loss_fn = nn.SmoothL1Loss()

    global_step = 0
    train_step = 0
    for episode in range(params.total_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # epsilon value updated based on the no of steps
            agent.epsilon = linear_epsilon(
                global_step, params.epsilon_start, params.epsilon_end,
                params.epsilon_decay_steps,
            )
            action = agent.choose_action(env.action_space, state, net)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            global_step += 1

            # the purpose of warmup_steps is to fill the buffer with states inorder to start learning
            if len(buffer) >= params.warmup_steps:
                states, actions, rewards, next_states, dones = buffer.sample(
                    params.batch_size
                )

                states = torch.from_numpy(states).to(device)
                actions = torch.from_numpy(actions).to(device)
                rewards = torch.from_numpy(rewards).to(device)
                next_states = torch.from_numpy(next_states).to(device)
                dones = torch.from_numpy(dones).to(device)

                # Q(s, a) for the actions actually taken
                predicted_q = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = target_net(next_states).max(dim=1)[0]
                    target_q = rewards + params.gamma * max_next_q * (1 - dones)

                loss = loss_fn(predicted_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_step += 1
                # for every target_sync_steps, we copy the weights of original into target
                if train_step % params.target_sync_steps == 0:
                    target_net.load_state_dict(net.state_dict())

        print(
            f"episode {episode}: reward={episode_reward}, "
            f"epsilon={agent.epsilon:.3f}, steps={global_step}"
        )

        if (episode + 1) % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"dqn_episode_{episode + 1}.pt")
            torch.save(net.state_dict(), path)
            print(f"saved checkpoint: {path}")

    final_path = os.path.join(checkpoint_dir, "dqn_final.pt")
    torch.save(net.state_dict(), final_path)
    print(f"saved final checkpoint: {final_path}")


if __name__ == "__main__":
    # small values here so this runs quickly as a smoke test; use main.py's
    # `params` (or similar modest values) for a real training run
    smoke_params = Params(
        total_episodes=2,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
        seed=123,
        batch_size=8,
        buffer_capacity=200,
        warmup_steps=20,
        target_sync_steps=1_000,
        num_frames=4,
        frame_skip=4,
    )
    train(smoke_params)
