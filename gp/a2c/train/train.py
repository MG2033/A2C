from gp.base.base_train import BaseTrainer
from gp.utils.utils import LearningRateDecay
from gp.utils.utils import create_experiment_dirs, create_list_dirs
import numpy as np
import time
from tqdm import tqdm
from gp.configs.a2c_config import A2CConfig
from gp.a2c.bench.env_summary_logger import EnvSummaryLogger


class Trainer(BaseTrainer):
    def __init__(self, sess, env, model, r_discount_factor=0.99,
                 lr_decay_method='linear', FLAGS=None):
        super().__init__(sess, model, None, A2CConfig, FLAGS)
        self.train_policy = self.model.train_policy
        self.step_policy = self.model.step_policy
        self.env = env
        self.sess = sess
        self.num_steps = self.model.num_steps
        self.cur_iteration = 0

        self.states = self.step_policy.initial_state
        self.dones = [False for _ in range(env.num_envs)]
        self.train_input_shape = (self.model.train_batch_size, self.model.img_height, self.model.img_width,
                                  self.model.num_classes * self.model.num_stack)

        self.observation_s = np.zeros(
            (env.num_envs, self.model.img_height, self.model.img_width, self.model.num_classes * self.model.num_stack),
            dtype=np.uint8)
        self.__observation_update(env.reset())

        self.gamma = r_discount_factor
        self.num_iterations = int(self.config.num_iterations)

        self.learning_rate_decayed = LearningRateDecay(v=self.config.learning_rate,
                                                       nvalues=self.num_iterations * self.config.unroll_time_steps * self.config.num_envs,
                                                       lr_decay_method=lr_decay_method)

        self.enviroments_summarizer = EnvSummaryLogger(sess,
                                                       create_list_dirs(A2CConfig.summary_dir, 'env', A2CConfig.num_envs),
                                                       self.summary_placeholders, self.summary_ops)

        self.summaries_arr_dict = [{} for _ in range(env.num_envs)]

    def train(self):
        tstart = time.time()
        loss_list = np.zeros(100, )
        i = 0
        for iteration in tqdm(range(self.global_step_tensor.eval(self.sess), self.num_iterations + 1, 1)):
            self.cur_iteration = iteration

            obs, states, rewards, masks, actions, values = self.__rollout()
            loss, policy_loss, value_loss, policy_entropy = self.__rollout_update(obs, states, rewards, masks, actions,
                                                                                  values)
            loss_list[i] = loss
            # TODO Tensorboard logging of policy_loss, value_loss, policy_entropy,...etc.
            nseconds = time.time() - tstart
            fps = int((iteration * self.num_steps * self.env.num_envs) / nseconds)

            # Update the Global step
            self.global_step_assign_op.eval(session=self.sess, feed_dict={
                self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})
            i += 1

            if not i % 100:
                mean_loss = np.mean(loss_list)
                print('Iteration:' + str(iteration) + '--loss:' + str(mean_loss))
                i = 0

            # Summary Helper is for all environments. Summary Writer is the main writer.
            self.enviroments_summarizer.add_summary_all(self.cur_iteration, self.summaries_arr_dict, summaries_merged=None)

        self.env.close()

    def __rollout_update(self, observations, states, rewards, masks, actions, values):
        # Updates the model per trajectory for using parallel environments. Uses the train_policy.
        advantages = rewards - values
        for step in range(len(observations)):
            current_learning_rate = self.learning_rate_decayed.value()
        feed_dict = {self.train_policy.X_input: observations, self.model.actions: actions,
                     self.model.advantage: advantages,
                     self.model.reward: rewards, self.model.learning_rate: current_learning_rate,
                     self.model.is_training: True}
        if states != []:
            # Leave it for now. It's for LSTM policy.
            feed_dict[self.model.S] = states
            feed_dict[self.model.M] = masks
        loss, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.model.loss, self.model.policy_gradient_loss, self.model.value_function_loss, self.model.entropy,
             self.model.optimize],
            feed_dict
        )
        return loss, policy_loss, value_loss, policy_entropy

    def __observation_update(self, observation):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        self.observation_s = np.roll(self.observation_s, shift=-1, axis=3)
        self.observation_s[:, :, :, -1] = observation[:, :, :, 0]

    def __discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        # Start from downwards to upwards like Bellman backup operation.
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]

    def __rollout(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states

        for n in range(self.num_steps):
            # Choose an action based on the current observation
            actions, values, states = self.step_policy.step(self.observation_s, self.states, self.dones)

            # Actions, Values predicted across all parallel environments
            mb_obs.append(np.copy(self.observation_s))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Take a step in the real environment
            observation, rewards, dones, _ = self.env.step(actions)
            # States and Masks are for LSTM Policy
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.observation_s[n] *= 0
            self.__observation_update(observation)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # Conversion from (time_steps, num_envs) to (num_envs, time_steps)
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.train_input_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.step_policy.value(self.observation_s, self.states, self.dones).tolist()

        # Discount/bootstrap off value fn in all parallel environments
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = self.__discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = self.__discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
            np_rewards = np.array(rewards)
            np_dones = np.array(dones)
            print(np_rewards[np_dones == 1])
            self.summaries_arr_dict[n]['reward'] = np.mean(np_rewards[np_dones == 1])

        # Instead of (num_envs, time_steps). Make them num_envs*time_steps.
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values
