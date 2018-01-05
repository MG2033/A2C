import pickle
from envs.subproc_vec_env import *
from models.model import Model
from train import Trainer
from utils.utils import set_all_global_seeds


class A2C:
    def __init__(self, sess, args):
        self.args = args
        self.model = Model(sess,
                           optimizer_params={
                               'learning_rate': args.learning_rate, 'alpha': 0.99, 'epsilon': 1e-5}, args=self.args)
        self.trainer = Trainer(sess, self.model, args=self.args)
        self.env_class = A2C.env_name_parser(self.args.env_class)

    def train(self):
        env = A2C.make_all_environments(self.args.num_envs, self.env_class, self.args.env_name,
                                        self.args.env_seed)

        print("\n\nBuilding the model...")
        self.model.build(env.observation_space.shape, env.action_space.n)
        print("Model is built successfully\n\n")

        with open(self.args.experiment_dir + self.args.env_name + '.pkl', 'wb') as f:
            pickle.dump((env.observation_space.shape, env.action_space.n), f, pickle.HIGHEST_PROTOCOL)

        print('Training...')
        try:
            # Produce video only if monitor method is implemented.
            try:
                if self.args.record_video_every != -1:
                    env.monitor(is_monitor=True, is_train=True, experiment_dir=self.args.experiment_dir,
                                record_video_every=self.args.record_video_every)
            except:
                pass
            self.trainer.train(env)
        except KeyboardInterrupt:
            print('Error occured..\n')
            self.trainer.save()
            env.close()

    def test(self, total_timesteps):
        observation_space_shape, action_space_n = None, None
        try:
            with open(self.args.experiment_dir + self.args.env_name + '.pkl', 'rb') as f:
                observation_space_shape, action_space_n = pickle.load(f)
        except:
            print(
                "Environment or checkpoint data not found. Make sure that env_data.pkl is present in the experiment by running training first.\n")
            exit(1)

        env = self.make_all_environments(num_envs=1, env_class=self.env_class, env_name=self.args.env_name,
                                         seed=self.args.env_seed)

        self.model.build(observation_space_shape, action_space_n)

        print('Testing...')
        try:
            # Produce video only if monitor method is implemented.
            try:
                if self.args.record_video_every != -1:
                    env.monitor(is_monitor=True, is_train=False, experiment_dir=self.args.experiment_dir,
                                record_video_every=self.args.record_video_every)
                else:
                    env.monitor(is_monitor=True, is_train=False, experiment_dir=self.args.experiment_dir,
                                record_video_every=20)
            except:
                pass
            self.trainer.test(total_timesteps=total_timesteps, env=env)
        except KeyboardInterrupt:
            print('Error occured..\n')
            env.close()

    def infer(self, observation):
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
    @staticmethod
    def __env_maker(env_class, env_name, i, seed):
        def __make_env():
            return env_class(env_name, i, seed)

        return __make_env

    @staticmethod
    def make_all_environments(num_envs=4, env_class=None, env_name="SpaceInvaders", seed=42):
        set_all_global_seeds(seed)

        return SubprocVecEnv(
            [A2C.__env_maker(env_class, env_name, i, seed) for i in range(num_envs)])

    @staticmethod
    def env_name_parser(env_name):
        from envs.gym_env import GymEnv
        envs_to_class = {'GymEnv': GymEnv}

        if env_name in envs_to_class:
            return envs_to_class[env_name]
        raise ValueError("There is no environment with this name. Make sure that the environment exists.")
