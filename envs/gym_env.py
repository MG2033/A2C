from envs.base_env import BaseEnv
from envs.atari_wrappers import wrap_deepmind
from envs.monitor import Monitor
import gym
from gym import wrappers


class GymEnv(BaseEnv):
    def __init__(self, env_name, id, seed):
        super().__init__(env_name, id)
        self.seed = seed
        self.make()
        # Get the inside of the wrappers!
        self.gym_env = self.env.env.env.env.env.env.env
        self.monitor = self.env.env.env.env.env.env.monitor

    def make(self):
        env = Monitor(gym.make(self.env_name), self.rank)
        env.seed(self.seed + self.rank)
        self.env = wrap_deepmind(env)
        return env

    def step(self, data):
        observation, reward, done, info = self.env.step(data)
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()

    def get_action_space(self):
        return self.env.action_space

    def get_observation_space(self):
        return self.env.observation_space

    def render(self):
        self.gym_env.render()
