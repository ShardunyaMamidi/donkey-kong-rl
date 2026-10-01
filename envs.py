import gymnasium as gym
import ale_py

from wrappers import FrameSkip, ResizeObservation, FrameStack, ClipReward


# the one place the env + wrapper chain is built, for training and playback
def make_env(cfg, render_mode=None):
    gym.register_envs(ale_py)
    env = gym.make("ALE/DonkeyKong-v5", obs_type="grayscale", render_mode=render_mode)
    env = FrameSkip(env, skip=cfg.frame_skip)
    env = ClipReward(env)
    env = ResizeObservation(env, size=(84, 84))
    env = FrameStack(env, num_frames=cfg.num_frames)
    return env
