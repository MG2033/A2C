from gp.a2c.envs.base_env import BaseEnv
from gp.a2c.envs.atari_wrappers import wrap_deepmind
import gym
from gym import wrappers


class GymEnv(BaseEnv):
    def __init__(self, env_name, id, seed):
        super().__init__(env_name, id)
        self.seed = seed
        self.make()

    def make(self):
        env = gym.make(self.env_name)
        env.seed(self.seed + self.rank)
        self.env = wrap_deepmind(env)
        return env

    def step(self, data):
        return self._info_step(self.env.step(data))

    def reset(self):
        return self._info_reset(self.env.reset())

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
