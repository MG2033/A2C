import tensorflow as tf
from gp.a2c.envs.gym_env import GymEnv
from gp.a2c.envs.subproc_vec_env import *
from gp.utils.utils import set_all_global_seeds
from gp.a2c.models.model import Model
from gp.a2c.models.cnn_policy import CNNPolicy
from gp.a2c.train.train import Trainer
from gp.configs.a2c_config import A2CConfig


class A2C:
    def __init__(self, num_envs, env_class=GymEnv, policy_class=CNNPolicy, num_steps=5, num_stack=4,
                 num_iterations=4e6, env_name="SpaceInvaders", learning_rate=7e-4, r_discount_factor=0.99,
                 seed=42, max_to_keep=10, experiment_dir="", is_train=True, cont_train=True):
        tf.reset_default_graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_envs,
                                inter_op_parallelism_threads=num_envs, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        self.env = self.__make_all_environments(num_envs, env_class, env_name, seed)
        self.model = Model(sess, policy_class, self.env.observation_space, self.env.action_space, num_envs,
                           num_steps, num_stack, total_timesteps=int(num_iterations * num_steps * num_envs),
                           optimizer_params={
                               'learning_rate': learning_rate, 'alpha': 0.99, 'epsilon': 1e-5})
        self.trainer = Trainer(sess, self.env, self.model, num_iterations, r_discount_factor=r_discount_factor,
                               max_to_keep=max_to_keep, is_train=is_train, cont_train=cont_train,
                               experiment_dir=experiment_dir)

    def train(self):
        print('Training...')
        self.trainer.train()

    def __make_all_environments(self, num_envs=4, env_class=GymEnv, env_name="SpaceInvaders", seed=42):
        set_all_global_seeds(seed)
        return SubprocVecEnv([env_class(env_name, i, seed) for i in range(num_envs)])


if __name__ == '__main__':
    a2c = A2C(A2CConfig.num_envs, env_class=A2CConfig.env_class, policy_class=A2CConfig.policy_class,
              num_steps=A2CConfig.unroll_time_steps,
              num_stack=A2CConfig.num_stack,
              num_iterations=A2CConfig.num_iterations, env_name=A2CConfig.env_name,
              seed=A2CConfig.env_seed, max_to_keep=A2CConfig.max_to_keep, experiment_dir=A2CConfig.experiment_dir,
              is_train=A2CConfig.is_train, cont_train=A2CConfig.cont_train,
              learning_rate=A2CConfig.learning_rate, r_discount_factor=A2CConfig.reward_discount_factor)

    a2c.train()
