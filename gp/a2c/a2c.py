import tensorflow as tf
from gp.a2c.envs.gym_env import GymEnv
from gp.a2c.envs.subproc_vec_env import *
from gp.utils.utils import set_all_global_seeds
from gp.a2c.models.model import Model
from gp.a2c.train.train import Trainer
from gp.configs.a2c_config import A2CConfig
import pickle


class A2C:
    def __init__(self, inference=True):
        tf.reset_default_graph()

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=A2CConfig.num_envs,
                                inter_op_parallelism_threads=A2CConfig.num_envs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        if not inference:
            self.env = self.make_all_environments(A2CConfig.num_envs, A2CConfig.env_class, A2CConfig.env_name,
                                                  A2CConfig.env_seed)

            self.model = Model(sess, self.env.observation_space.shape, self.env.action_space.n,
                               optimizer_params={
                                   'learning_rate': A2CConfig.learning_rate, 'alpha': 0.99, 'epsilon': 1e-5})

            with open(A2CConfig.experiment_dir + 'env_data.pkl', 'wb') as f:
                pickle.dump((self.env.observation_space.shape, self.env.action_space.n), f, pickle.HIGHEST_PROTOCOL)

            print("\n\nBuilding the model...")
            self.model.build()
            print("Model is built successfully\n\n")

            self.trainer = Trainer(sess, self.env, self.model)
            self.train = self.__train

        else:
            try:
                with open(A2CConfig.experiment_dir + 'env_data.pkl', 'rb') as f:
                    observation_space_shape, action_space_n = pickle.load(f)

            except:
                print("Environment or checkpoint data not found. Make sure that env_data.pkl is present in the experiment")
                exit(1)

            self.model = Model(sess, observation_space_shape, action_space_n,
                               optimizer_params={
                                   'learning_rate': A2CConfig.learning_rate, 'alpha': 0.99, 'epsilon': 1e-5})

            print("\n\nBuilding the model...")
            self.model.build()
            print("Model is built successfully\n\n")
            print(A2CConfig.checkpoint_dir)
            latest_checkpoint = tf.train.latest_checkpoint(A2CConfig.checkpoint_dir)
            self.saver = tf.train.Saver(max_to_keep=A2CConfig.max_to_keep)
            if latest_checkpoint:
                print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)
                print("Model loaded")

            self.test = self.__test
            self.infer = self.__infer

    def __train(self):
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

    def __test(self, total_timesteps):
        print('Testing...')
        try:
            env = self.make_all_environments(num_envs=1, env_class=GymEnv, env_name=A2CConfig.env_name,
                                             seed=A2CConfig.env_seed)
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

    def __infer(self, observation):
        """Used for inference.
        :param observation: (tf.tensor) having the shape (None,img_height,img_width,num_classes*num_stack)
        :return action after noise and argmax
        :return value function of the state
        """
        states = self.model.step_policy.initial_state
        dones = []
        # states and dones are for LSTM, leave them for now!
        action, value, states = self.model.step_policy.step(observation, states, dones)
        return action, value

    # The reason behind this design pattern is to pass the function handler when required after serialization.
    def __env_maker(self, env_class, env_name, i, seed):
        def __make_env():
            return env_class(env_name, i, seed)

        return __make_env

    def make_all_environments(self, num_envs=4, env_class=GymEnv, env_name="SpaceInvaders", seed=42):
        set_all_global_seeds(seed)

        return SubprocVecEnv(
            [self.__env_maker(env_class, env_name, i, seed) for i in range(num_envs)])


if __name__ == '__main__':
    a2c = A2C(inference=False)
    # a2c.train()
    # a2c.test(total_timesteps=10000000)
