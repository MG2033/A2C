import tensorflow as tf
from gp.a2c.envs.gym_env import GymEnv
from gp.a2c.envs.subproc_vec_env import *
from gp.utils.utils import set_all_global_seeds
from gp.a2c.models.model import Model
from gp.a2c.train.train import Trainer
from gp.configs.a2c_config import A2CConfig

class A2C:
    def __init__(self):
        tf.reset_default_graph()

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=A2CConfig.num_envs,
                                inter_op_parallelism_threads=A2CConfig.num_envs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        self.env = self.__make_all_environments(A2CConfig.num_envs, A2CConfig.env_class, A2CConfig.env_name,
                                                A2CConfig.env_seed)
        self.model = Model(sess, self.env.observation_space, self.env.action_space,
                           optimizer_params={
                               'learning_rate': A2CConfig.learning_rate, 'alpha': 0.99, 'epsilon': 1e-5})

        print("\n\nBuilding the model...")
        self.model.build()
        print("Model is built successfully\n\n")

        self.trainer = Trainer(sess, self.env, self.model)

    def train(self):
        print('Training...')
        try:
            if A2CConfig.record_video_every != -1:
                self.env.monitor(is_monitor=True, is_train=True, experiment_dir=A2CConfig.experiment_dir,
                                 record_video_every=A2CConfig.record_video_every)
            self.trainer.train()
        except KeyboardInterrupt:
            print('Error occured..')
            self.trainer.save()
            self.env.close()

    def test(self, total_timesteps):
        print('Testing...')
        try:
            env = self.__make_all_environments(num_envs=1, env_class=GymEnv, env_name=A2CConfig.env_name, seed=A2CConfig.env_seed)
            if A2CConfig.record_video_every != -1:
                env.monitor(is_monitor=True, is_train=False, experiment_dir=A2CConfig.experiment_dir,
                                 record_video_every=A2CConfig.record_video_every)
            else:
                env.monitor(is_monitor=True, is_train=False, experiment_dir=A2CConfig.experiment_dir,
                                 record_video_every=20)
            self.trainer.test(total_timesteps=total_timesteps, env=env)
        except KeyboardInterrupt:
            print('Error occured..')
            self.env.close()

    # The reason behind this design pattern is to pass the function handler when required after serialization.
    def __env_maker(self, env_class, env_name, i, seed):
        def __make_env():
            return env_class(env_name, i, seed)

        return __make_env

    def __make_all_environments(self, num_envs=4, env_class=GymEnv, env_name="SpaceInvaders", seed=42):
        set_all_global_seeds(seed)

        return SubprocVecEnv(
            [self.__env_maker(env_class, env_name, i, seed) for i in range(num_envs)])


if __name__ == '__main__':
    a2c = A2C()
    a2c.train()
    #a2c.test(total_timesteps=10000000)
