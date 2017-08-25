class BaseEnv:
    def __init__(self, env_name, id):
        self.env_name = env_name
        self.rank = id
        self.env = None
        self.rewards = []
        self.summaries_dict = {'reward': 0, 'episode_length': 0}
        self.episode_length = 0

    def make(self):
        raise NotImplementedError("make method is not implemented")

    def step(self, data):
        raise NotImplementedError("step method is not implemented")

    def reset(self):
        raise NotImplementedError("reset method is not implemented")

    def get_action_space(self):
        raise NotImplementedError("get_action_space method is not implemented")

    def get_observation_space(self):
        raise NotImplementedError("get_observation_space method is not implemented")

    def _monitor_step(self, state):
        observation, reward, done, info = state
        self.rewards.append(reward)
        self.episode_length += 1
        self.summaries_dict['reward'] = -1
        if done:
            self.summaries_dict['reward'] = sum(self.rewards)
            self.summaries_dict['episode_length'] = self.episode_length
            self.rewards = []
            self.episode_length = 0
        return state

    def _monitor_reset(self, state):
        self.rewards = []
        self.episode_length = 0
        return state
