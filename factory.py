from agents.dqn import DQNAgent


# algorithm name -> agent class; adding an algorithm (e.g. PPO) means
# implementing BaseAgent and adding one entry here
AGENTS = {
    DQNAgent.name: DQNAgent,
}


def get_agent_cls(algo):
    if algo not in AGENTS:
        raise ValueError(f"unknown algo {algo!r}, choose from {sorted(AGENTS)}")
    return AGENTS[algo]


# builds the algorithm's config class; unspecified fields keep their defaults
def make_config(algo, **overrides):
    return get_agent_cls(algo).config_cls(**overrides)


def make_agent(algo, cfg, env, device):
    return get_agent_cls(algo)(cfg, env.observation_space, env.action_space, device)
