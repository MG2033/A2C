import tensorflow as tf
from gp.i2a.rollout_encoder.rollouts_encoder import RolloutsEncoder
from gp.i2a.model_free_path.a2c_cnn import A2CCNN
from gp.i2a.imagination_core.rollout_policy import RolloutPolicy
from gp.i2a.imagination_core.rollout import Rollout
from gp.i2a.model_free_path.a2c_cnn import A2CCNN


class I2a:
    def __init__(self, config):
        self.config = config
        with tf.name_scope('i2a_input'):
            self.input = tf.placeholder(tf.float32, self.config.state_size,
                                        name='states')

        self.initial_lstm_state = tf.placeholder(tf.float32, [2, None, self.config.lstm_size],
                                                 name='lstm_initial_state')

    def actor_critic(self, rollout_encoding, model_free_feature_map):
        input = tf.concat(rollout_encoding, model_free_feature_map)
        h1 = tf.layers.dense(input, 512, kernel_initializer=tf.contrib.layers.xavier_initializer())
        h2 = tf.layers.dense(h1, 256, kernel_initializer=tf.contrib.layers.xavier_initializer())
        output = tf.layers.dense(h2, self.config.actions_num + 1,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        return output[:-1], output[-1]

    def build_model(self):
        with tf.name_scope('i2a_model'):
            rollout = Rollout(self.input, self.initial_lstm_state, self.config)
            rollout_encoder = RolloutsEncoder(rollout.imagined_observations, rollout.imagined_rewards, self.config)
            a2c_cnn = A2CCNN(self.input)
            model_free_feature_map = a2c_cnn.cnn_output()
            policy, v = self.actor_critic(rollout_encoder.rollout_encoding, model_free_feature_map)
