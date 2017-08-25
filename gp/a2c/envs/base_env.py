class BaseEnv:
    def __init__(self, env_name, id):
        self.env_name = env_name
        self.rank = id
        self.env = None
        self.rewards = []
        self.summaries_dict = {'reward': 0}

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
        if done:
            self.summaries_dict['reward'] = sum(self.rewards)
            self.rewards = [reward]
        return state

    def _monitor_reset(self, state):
        self.rewards = [0]
        self.summaries_dict['reward'] = sum(self.rewards)
        return state
