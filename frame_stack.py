from collections import deque

import gymnasium as gym
import numpy as np

from robot_particle_env import RobotParticleEnv


class StackedRobotParticleEnv(gym.Env):
    def __init__(self, env_config):
        self.env = FrameStack(RobotParticleEnv(env_config), num_stack=4)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)

        # Update observation space
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(observation)
        return self._get_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return np.array(self.frames)
