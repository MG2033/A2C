import tensorflow as tf
import gym
from gp.a2c.envs.gym_env import GymEnv
from gp.a2c.envs.subproc_vec_env import *
from gp.utils.utils import set_all_global_seeds


class a2c:
    def __init__(self, num_envs, env_class=GymEnv, env_name="SpaceInvaders", seed=42):
        self.num_envs = num_envs
        self.env_class = env_class
        self.env_name = env_name
        self.seed = seed
        self.envs = None

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_envs,
                                inter_op_parallelism_threads=num_envs, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        self.envs = self.__make_all_environments(self.num_envs, self.env_class, self.env_name, self.seed)

    def __make_all_environments(self, num_envs=4, env_class=GymEnv, env_name="SpaceInvaders", seed=42):
        set_all_global_seeds(seed)
        return SubprocVecEnv([env_class(env_name, i, seed) for i in range(num_envs)])

    def build_model(self):
        pass


if __name__ == '__main__':
    num_envs = 5
    env_class = GymEnv
    env_name = "SpaceInvaders"
    env_seed = 42
    a2c = a2c(num_envs, env_class, env_name, env_seed)
