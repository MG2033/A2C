import time
from gym import Wrapper


class Monitor(Wrapper):
    def __init__(self, env, rank=0):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.rank = rank
        self.rewards = []
        self.total_steps = 0
        self.current_metadata = {}  # extra info that gets injected into each log entry
        self.summaries_dict = {'reward': 0, 'episode_length': 0, 'step': 0}
        self.tstart = time.time()

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
            self.summaries_dict['step'] = round(time.time() - self.tstart)
        self.total_steps += 1
        info = self.summaries_dict
        return observation, reward, done, info

    def get_summaries_dict(self):
        return self.summaries_dict
