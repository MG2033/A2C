import time


class BaseEnv:
    def __init__(self, env_name, id):
        self.env_name = env_name
        self.rank = id
        self.env = None

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

    def monitor(self, is_monitor, is_train, video_record_dir="", record_video_every=10):
        raise NotImplementedError("monitor method is not implemented")

    def render(self):
        raise NotImplementedError("render method is not implemented")

