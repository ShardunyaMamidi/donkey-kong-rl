import gymnasium as gym
import numpy as np
from collections import deque
import cv2

# used for frameskipping, we are considering every 4th frames and only considering last two which are then max pooled and returned
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip  # skips 4 frames

    def step(self, action):
        total_reward = 0.0
        last_two = deque(maxlen=2)
        terminated = truncated = False

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            last_two.append(obs)
            total_reward += reward
            if terminated or truncated:
                break

        # maxpooling the two frames to ensure all sprites are caught
        max_frame = np.maximum(last_two[-2], last_two[-1]) if len(last_two) == 2 else last_two[-1]
        return max_frame, total_reward, terminated, truncated, info


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, size=(84, 84)):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=size, dtype=np.uint8
        )

    # resizing the greyscale image from 210x160 to 84x84
    def observation(self, observation):
        return cv2.resize(observation, self.size, interpolation=cv2.INTER_AREA)


# creating a queue that stack 4 frames to determine the trajectories in the game
class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.num_frames = num_frames
        # using a queue for FIFO
        self.frames = deque(maxlen=num_frames)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(num_frames, *old_shape), dtype=np.uint8
        )

    # overriding reset method here as we are considering the 4 latest frames for any input
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0), info

    # called automatically and the frames are appended during each step()
    def observation(self, observation):
        self.frames.append(observation)
        return np.stack(self.frames, axis=0)


class ClipReward(gym.RewardWrapper):
    # called automatically on the reward returned by env.step()
    def reward(self, reward):
        # either -1, 0, 1
        return np.sign(reward)
