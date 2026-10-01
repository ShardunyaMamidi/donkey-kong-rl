import argparse

from factory import AGENTS, make_config
from trainer import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=sorted(AGENTS), default="dqn")
    args = parser.parse_args()

    # real training run using the algorithm config's default hyperparameters
    train(args.algo, make_config(args.algo))
