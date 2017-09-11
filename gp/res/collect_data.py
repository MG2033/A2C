from gp.utils.utils import create_dirs
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import gym
from gp.a2c.a2c import A2C

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('save_dir', "/shared/oabdelta/new-torcs-data/", """ directory to save to """)
tf.app.flags.DEFINE_integer('episodes', 50, """ number of episodes """)
tf.app.flags.DEFINE_integer('episode_len', 45, """ number of episode steps """)
tf.app.flags.DEFINE_integer('max_episode_len', 500, """ number of episode steps """)


class Collector:
    def __init__(self, env_id, policy):
        create_dirs([FLAGS.save_dir])
        self.env = gym.make(env_id)
        self.action_dims = self.env.action_space.n
        self.state_size = self.env.observation_space.shape

        self.states = np.zeros((FLAGS.episodes, FLAGS.episode_len + 1) + self.state_size)
        self.rewards = np.zeros((FLAGS.episodes, FLAGS.episode_len))
        self.actions = np.zeros((FLAGS.episodes, FLAGS.episode_len))
        self.action_space = np.arange(self.action_dims)

        self.policy = policy

    def collect_data(self):
        ob = self.env.reset()

        epsd = 0
        while epsd < FLAGS.episodes:
            states = np.zeros((FLAGS.max_episode_len + 1,) + self.state_size)
            rewards = np.zeros((FLAGS.max_episode_len))
            actions = np.zeros((FLAGS.max_episode_len))
            print('episode: ', epsd)
            for step in tqdm(range(FLAGS.max_episode_len)):
                policy_action = self.policy(ob)
                action = self.action_space[policy_action]

                # print(action)
                self.env.render()

                ob, reward, done, _ = self.env.step([action])
                rewards[step] = reward
                # print(reward)
                actions[step] = action

                if done or step == FLAGS.max_episode_len - 1:
                    states[step + 1, :] = ob.track

                    for i in range(int(step / FLAGS.episode_len)):
                        self.states[epsd] = states[step - (i + 1) * FLAGS.episode_len:step - i * FLAGS.episode_len + 1]
                        self.actions[epsd] = actions[step - (i + 1) * FLAGS.episode_len:step - i * FLAGS.episode_len]
                        self.rewards[epsd] = rewards[step - (i + 1) * FLAGS.episode_len:step - i * FLAGS.episode_len]
                        epsd += 1

                    ob = self.env.reset()
                    break

                self.save()

    def save(self):
        np.save(FLAGS.save_dir + 'states.npy', self.states)
        np.save(FLAGS.save_dir + 'actions.npy', self.actions)
        np.save(FLAGS.save_dir + 'rewards.npy', self.rewards)


def main(_):
    env_id = 'PongNoFrameskip-v4'
    a2c = A2C()
    data_collector = Collector(env_id, a2c.infer)

    data_collector.collect_data()

    data_collector.save()


if __name__ == '__main__':
    tf.app.run()
