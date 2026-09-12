import gymnasium as gym
import ale_py
import numpy as np
from typing import NamedTuple

gym.register_envs(ale_py)
seed = np.random.default_rng(123)

# used to set up the parametes of the envi and model
class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float  # discount rate
    epsilon: float  # for exploration/exploitation
    seed: int  # for a defined randomness
    is_slippery: bool  #
    n_runs: int
    action_size: int  # number of possible actions
    state_size: int  # no of possible states

params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
)

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

env = gym.make(
    "ALE/DonkeyKong-v5",
    render_mode="rgb_array"
)

# Setting the action and state space of donkeykong
params = params._replace(action_size=env.action_space.shape)
params = params._replace(state_size=env.observation_space.shape)

print(f"action size: {params.action_size}, state size: {params.state_size}")
