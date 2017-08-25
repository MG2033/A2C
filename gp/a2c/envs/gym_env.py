from gp.a2c.envs.base_env import BaseEnv
from gp.a2c.envs.atari_wrappers import wrap_deepmind
import gym
from gym import wrappers


class GymEnv(BaseEnv):
    def __init__(self, env_name, id, seed, record=False, video_record_dir="", record_video_every=10):
        super().__init__(env_name, id)
        self.seed = seed
        self.make()
        if record:
            self.env = wrappers.Monitor(self.env, video_record_dir, resume=True,
                                        video_callable=lambda count: count % record_video_every == 0)

    def make(self):
        env = gym.make(self.env_name)
        env.seed(self.seed + self.rank)
        self.env = wrap_deepmind(env)
        return env

    def step(self, data):
        return self._monitor_step(self.env.step(data))

    def reset(self):
        return self._monitor_reset(self.env.reset())

    def get_action_space(self):
        return self.env.action_space

    def get_observation_space(self):
        return self.env.observation_space
