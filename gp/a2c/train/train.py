from gp.a2c.train.base_train import BaseTrainer
import numpy as np
import time


class Trainer(BaseTrainer):
    def __init__(self, sess, env, model, num_iterations, r_discount_factor, max_to_keep, experiment_dir, is_train,
                 cont_train):
        BaseTrainer.__init__(sess, model, max_to_keep, experiment_dir, is_train, cont_train)
        self.observation_s = None
        self.__observation_update(env.reset())
        self.model = model
        self.sess = sess
        self.num_steps = self.model.num_steps
        self.num_stack = self.model.num_stack

        self.states = self.model.initial_state
        self.dones = [False for _ in range(env.num_envs)]
        self.input_shape = (self.model.batch_size, self.model.img_height, self.model.img_width,
                            self.model.num_classes * self.model.num_stack)
        self.env = env
        self.gamma = r_discount_factor
        self.num_iterations = num_iterations

        print("\n\nBuilding the model...")
        self.build_model()
        print("Model is built successfully\n\n")

    def build_model(self):
        # Build the neural network
        self.model.init_input()
        self.model.init_network()

    def train(self):
        tstart = time.time()
        for iteration in range(self.model.global_step_tensor.eval(self.sess), self.num_iterations + 1, 1):
            obs, states, rewards, masks, actions, values = self.__rollout()
            policy_loss, value_loss, policy_entropy = self.__rollout_update(obs, states, rewards, masks, actions,
                                                                            values)
            # TODO Tensorboard logging of policy_loss, value_loss, policy_entropy,...etc.
            print('Iteration ' + str(iteration) + '.')
            nseconds = time.time() - tstart
            fps = int((iteration * self.num_steps * self.env.num_envs) / nseconds)

            # Update the Global step
            self.model.global_step_assign_op.eval(session=self.sess, feed_dict={
                self.model.global_step_input: self.model.global_step_tensor.eval(self.sess) + 1})
        self.env.close()

    def __rollout_update(self, observations, states, rewards, masks, actions, values):
        # Updates the model per trajectory for using parallel environments
        advantages = rewards - values
        for step in range(len(observations)):
            current_learning_rate = self.model.learning_rate.value()
        feed_dict = {self.model.X_input: observations, self.model.gt_actions: actions, self.model.advantage: advantages,
                     self.model.reward: rewards, self.model.learning_rate: current_learning_rate}
        if states != []:
            feed_dict[self.model.S] = states
            feed_dict[self.model.M] = masks
        policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.model.policy_gradient_loss, self.model.value_function_loss, self.model.entropy, self.model.optimize],
            feed_dict
        )
        return policy_loss, value_loss, policy_entropy

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
            actions, values, states = self.model.step(self.observation_s, self.states, self.dones)

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
                    self.observation_s[n] = self.observation_s[n] * 0
            self.__observation_update(observation)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # Conversion from (time_steps, num_envs) to (num_envs, time_steps)
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.input_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.observation_s, self.states, self.dones).tolist()

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
