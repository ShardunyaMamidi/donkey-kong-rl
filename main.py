from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sns.set_theme()

# used to set up the parametes of the envi and model
class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float  # discount rate
    epsilon: float  # for exploration/exploitation
    map_size: int  # no of tiles in the environment
    seed: int  # for a defined randomness
    is_slippery: bool  #
    n_runs: int
    action_size: int  # number of possible actions
    state_size: int  # no of possible states
    proba_frozen: float  # prob that a tile is frozen

params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9
)

# random generator
seed = np.random.default_rng(params.seed)

# Frozen Lake env
env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    )
)

# Setting the action and state space based on the env (frozen lake)
params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)

# Q learning and impl
class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        # resets the q-table
        self.reset_qtable()

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

    # method to update the newer q-value in the q-table
    def update(self, state, action, reward, new_state):
        # update logic = Q(s,a) = Q(s,a) + lr * [R(s,a) + gamma * max( Q(s`, a`) ) - Q(s, a)]
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )

        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

# epsilon-greedy
class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    # exploration vs exploitation
    def choose_action(self, action_space, state, q_table):
        explor_exploit_prob = seed.uniform(0, 1)

        # exploration
        if explor_exploit_prob < self.epsilon:
            action = action_space.sample()

        # exploitation
        else:
            # choose action where q-value is max
            max_ids = np.where(q_table[state, :] == max(q_table[state, :]))[0]
            # if there are more than one with highest q-value, choose random from those
            action =  seed.choice(max_ids)
        return action

learner = Qlearning(
    learning_rate=params.learning_rate,
    gamma = params.gamma,
    state_size=params.state_size,
    action_size=params.action_size
)

explorer = EpsilonGreedy(
    epsilon=params.epsilon
)

# method to run/train the agent
def run_env():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states, all_actions = [], []

    for run in range(params.n_runs):
        learner.reset_qtable()  # resets the qtable after every run

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)
            step = 0
            done = False
            total_reward = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, q_table=learner.qtable,
                )

                all_states.append(state)
                all_actions.append(action)

                # Apply the action
                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_reward += reward
                step += 1

                state = new_state

            rewards[episode, run] = total_reward
            steps[episode, run] = step

        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_actions, all_states
