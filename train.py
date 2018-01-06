import time

import numpy as np
from tqdm import tqdm

from base_train import BaseTrainer
from envs.env_summary_logger import EnvSummaryLogger
from utils.lr_decay import LearningRateDecay
from utils.utils import create_list_dirs


class Trainer(BaseTrainer):
    def __init__(self, sess, model, r_discount_factor=0.99,
                 lr_decay_method='linear', args=None):
        super().__init__(sess, model, args)
        self.save_every = 20000
        self.sess = sess
        self.num_steps = self.model.num_steps
        self.cur_iteration = 0
        self.global_time_step = 0
        self.observation_s = None
        self.states = None
        self.dones = None
        self.env = None

        self.num_iterations = int(self.args.num_iterations)

        self.gamma = r_discount_factor

        self.learning_rate_decayed = LearningRateDecay(v=self.args.learning_rate,
                                                       nvalues=self.num_iterations * self.args.unroll_time_steps * self.args.num_envs,
                                                       lr_decay_method=lr_decay_method)

        self.env_summary_logger = EnvSummaryLogger(sess,
                                                   create_list_dirs(self.args.summary_dir, 'env', self.args.num_envs))

    def train(self, env):
        self._init_model()
        self._load_model()

        self.env = env
        self.observation_s = np.zeros(
            (env.num_envs, self.model.img_height, self.model.img_width, self.model.num_classes * self.model.num_stack),
            dtype=np.uint8)
        self.observation_s = self.__observation_update(self.env.reset(), self.observation_s)

        self.states = self.model.step_policy.initial_state
        self.dones = [False for _ in range(self.env.num_envs)]

        tstart = time.time()
        loss_list = np.zeros(100, )
        policy_entropy_list = np.zeros(100, )
        fps_list = np.zeros(100, )
        arr_idx = 0
        start_iteration = self.global_step_tensor.eval(self.sess)
        self.global_time_step = self.global_time_step_tensor.eval(self.sess)

        for iteration in tqdm(range(start_iteration, self.num_iterations + 1, 1), initial=start_iteration,
                              total=self.num_iterations):

            self.cur_iteration = iteration

            obs, states, rewards, masks, actions, values = self.__rollout()
            loss, policy_loss, value_loss, policy_entropy = self.__rollout_update(obs, states, rewards, masks, actions,
                                                                                  values)

            # Calculate and Summarize
            loss_list[arr_idx] = loss
            nseconds = time.time() - tstart
            fps_list[arr_idx] = int((iteration * self.num_steps * self.env.num_envs) / nseconds)
            policy_entropy_list[arr_idx] = policy_entropy

            # Update the Global step
            self.global_step_assign_op.eval(session=self.sess, feed_dict={
                self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})
            arr_idx += 1

            if not arr_idx % 100:
                mean_loss = np.mean(loss_list)
                mean_fps = np.mean(fps_list)
                mean_pe = np.mean(policy_entropy_list)
                print('Iteration:' + str(iteration) + ' - loss: ' + str(mean_loss)[:8] + ' - policy_entropy: ' + str(
                    mean_pe)[:8] + ' - fps: ' + str(mean_fps))
                arr_idx = 0
            if iteration % self.save_every == 0:
                self.save()
        self.env.close()

    def test(self, total_timesteps, env):
        self._init_model()
        self._load_model()

        states = self.model.step_policy.initial_state

        dones = [False for _ in range(env.num_envs)]

        observation_s = np.zeros(
            (env.num_envs, self.model.img_height, self.model.img_width,
             self.model.num_classes * self.model.num_stack),
            dtype=np.uint8)
        observation_s = self.__observation_update(env.reset(), observation_s)

        for _ in tqdm(range(total_timesteps)):
            actions, values, states = self.model.step_policy.step(observation_s, states, dones)
            observation, rewards, dones, _ = env.step(actions)
            for n, done in enumerate(dones):
                if done:
                    observation_s[n] *= 0
            observation_s = self.__observation_update(observation, observation_s)
        env.close()

    def __rollout_update(self, observations, states, rewards, masks, actions, values):
        # Updates the model per trajectory for using parallel environments. Uses the train_policy.
        advantages = rewards - values
        for step in range(len(observations)):
            current_learning_rate = self.learning_rate_decayed.value()
        feed_dict = {self.model.train_policy.X_input: observations, self.model.actions: actions,
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

    def __observation_update(self, new_observation, old_observation):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        updated_observation = np.roll(old_observation, shift=-1, axis=3)
        updated_observation[:, :, :, -1] = new_observation[:, :, :, 0]
        return updated_observation

    def __discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        # Start from downwards to upwards like Bellman backup operation.
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]

    def __rollout(self):
        train_input_shape = (self.model.train_batch_size, self.model.img_height, self.model.img_width,
                             self.model.num_classes * self.model.num_stack)

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states

        for n in range(self.num_steps):
            # Choose an action based on the current observation
            actions, values, states = self.model.step_policy.step(self.observation_s, self.states, self.dones)

            # Actions, Values predicted across all parallel environments
            mb_obs.append(np.copy(self.observation_s))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Take a step in the real environment
            observation, rewards, dones, info = self.env.step(actions)
            # plt.imsave(fname="img" + str(n) + ".png", arr=observation[0, :, :, 0], cmap='gray')

            # Tensorboard dump, divided by 100 to rescale (to make the steps make sense)
            self.env_summary_logger.add_summary_all(int(self.global_time_step / 100), info)
            self.global_time_step += 1
            self.global_time_step_assign_op.eval(session=self.sess, feed_dict={
                self.global_time_step_input: self.global_time_step})

            # States and Masks are for LSTM Policy
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.observation_s[n] *= 0
            self.observation_s = self.__observation_update(observation, self.observation_s)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # Conversion from (time_steps, num_envs) to (num_envs, time_steps)
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(train_input_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.step_policy.value(self.observation_s, self.states, self.dones).tolist()

        # Discount/bootstrap off value fn in all parallel environments
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = self.__discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = self.__discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        # Instead of (num_envs, time_steps). Make them num_envs*time_steps.
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values
