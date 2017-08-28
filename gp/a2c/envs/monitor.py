import time
from gym import Wrapper, wrappers


class Monitor(Wrapper):
    def __init__(self, env, rank=0):
        Wrapper.__init__(self, env=env)
        self.rank = rank
        self.rewards = []
        self.current_metadata = {}  # extra info that gets injected into each log entry
        self.summaries_dict = {'reward': 0, 'episode_length': 0}

    def reset(self):
        self.summaries_dict['reward'] = -1
        self.summaries_dict['episode_length'] = -1
        self.rewards = []
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.summaries_dict['reward'] = sum(self.rewards)
            self.summaries_dict['episode_length'] = len(self.rewards)
        info = self.summaries_dict
        return observation, reward, done, info

    def get_summaries_dict(self):
        return self.summaries_dict

    def monitor(self, is_monitor, is_train, experiment_dir="", record_video_every=10):
        if is_monitor:
            if is_train:
                self.env = wrappers.Monitor(self.env, experiment_dir + 'output', resume=True,
                                                video_callable=lambda count: count % record_video_every == 0)
            else:
                self.env = wrappers.Monitor(self.env, experiment_dir + 'test', resume=True,
                                                video_callable=lambda count: count % record_video_every == 0)
        else:
            self.env = wrappers.Monitor(self.env, experiment_dir + 'output', resume=True,
                                            video_callable=False)
        self.env.reset()
