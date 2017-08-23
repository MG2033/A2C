import numpy as np
class GenerateData:
    def __init__(self, config):
        np.random.seed(2)
        x = np.load(config.x_path)
        self._rewards = np.load(config.rewards_path)

        x = x[:2400]
        np.random.shuffle(x)
        self._y = x[:, 1:, :]
        self._x = x[:, :-1, :]
        self._rewards = self._rewards[:2400]
        np.random.shuffle(self._rewards)
        self._actions = None
        self._config = config

        self.prepare_actions()
        self.prepare_rewards()

        self._rewards = np.expand_dims(self._rewards, axis=2)

        self.xtest = self._x[self._config.num_episodes_train:config.num_episodes_train + config.num_episodes_test]
        self._x = self._x[:self._config.num_episodes_train]
        self.ytest = self._y[self._config.num_episodes_train:config.num_episodes_train + config.num_episodes_test]
        self._y = self._y[:self._config.num_episodes_train]
        self.actionstest = self._actions[
                           self._config.num_episodes_train:config.num_episodes_train + config.num_episodes_test]
        self._actions = self._actions[:self._config.num_episodes_train]
        self.rewardstest = self._rewards[
                           self._config.num_episodes_train:config.num_episodes_train + config.num_episodes_test]
        self._rewards = self._rewards[:self._config.num_episodes_train]

    def next_batch(self):
        while True:
            idx = np.random.choice(self._config.num_episodes_train, self._config.batch_size)
            self.current_x = self._x[idx]
            self.current_y = self._y[idx]
            self.current_actions = self._actions[idx]
            self.current_rewards = self._rewards[idx]
            for i in range(0, self._config.all_seq_length, self._config.truncated_time_steps):
                if i == 0:
                    new_sequence = True
                else:
                    new_sequence = False
                batch_x = self.current_x[:, i:i + self._config.truncated_time_steps, :]
                batch_y = self.current_y[:, i:i + self._config.truncated_time_steps, :]
                batch_actions = self.current_actions[:, i:i + self._config.truncated_time_steps, :]
                batch_rewards = self.current_rewards[:, i:i + self._config.truncated_time_steps, :]
                yield batch_x, batch_y, batch_actions, batch_rewards, new_sequence

    def prepare_rewards(self):
        self._rewards[self._rewards > 0] = 1
        self._rewards[self._rewards < 0] = -1

    def prepare_actions(self):
        self._actions = np.zeros(
            (self._config.num_episodes, self._config.all_seq_length, self._config.action_dim))
        difference = np.zeros((self._config.num_episodes, self._config.all_seq_length))
        actions = np.load(self._config.actions_path)
        actions = actions[:2400]
        np.random.shuffle(actions)

        # actions = actions[:16, :129]
        difference[:, 1:] = actions[:, 1:] - actions[:, :-1]

        for i in range(actions.shape[0]):
            self._actions[i, difference[i] > self._config.epsilon] = [0, 1, 0]
            self._actions[i, difference[i] < -1 * self._config.epsilon] = [0, 0, 1]
            self._actions[
                i, (difference[i] > -1 * self._config.epsilon) & (difference[i] < 1 * self._config.epsilon)] = [1, 0, 0]

