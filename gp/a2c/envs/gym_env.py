from gp.a2c.envs.base_env import BaseEnv
from gp.a2c.envs.atari_wrappers import wrap_deepmind
import gym


class GymEnv(BaseEnv):
    def __init__(self, env_name, id, seed):
        super().__init__(env_name, id)
        self.seed = seed
        self.make()

    def make(self):
        env = gym.make('{}NoFrameskip-v4'.format(self.env_name))
        env.seed(self.seed + self.rank)
        # TODO Logging
        self.env = wrap_deepmind(env)
        return env

    def step(self, data):
        return self.env.step(data)

    def reset(self):
        return self.env.reset()

    def get_action_space(self):
        return self.env.action_space

    def get_observation_space(self):
        return self.env.observation_space
