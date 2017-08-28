from gp.a2c.envs.base_env import BaseEnv
from gp.a2c.envs.atari_wrappers import wrap_deepmind
from gp.a2c.envs.monitor import Monitor
import gym
from gym import wrappers


class GymEnv(BaseEnv):
    def __init__(self, env_name, id, seed):
        super().__init__(env_name, id)
        self.seed = seed
        self.make()

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

    def monitor(self, is_monitor, is_train, experiment_dir="", record_video_every=10):
        if is_monitor:
            if is_train:
                self.env = wrappers.Monitor(self.env, experiment_dir + 'output', resume=True,
                                            video_callable=lambda count: count % record_video_every == 0)
            else:
                self.env = wrappers.Monitor(self.env, experiment_dir + 'test', resume=True,
                                            video_callable=lambda count: count % record_video_every == 0)
        else:
            self.env = wrappers.Monitor(self.env, experiment_dir + 'output', resume=True,
                                        video_callable=False)
        self.env.reset()
