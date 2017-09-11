import tensorflow as tf
from gp.i2a.rollout_encoder.rollouts_encoder import RolloutsEncoder
from gp.i2a.model_free_path.a2c_cnn import A2CCNN
from gp.i2a.imagination_core.rollout_policy import RolloutPolicy
from gp.i2a.imagination_core.rollout import Rollout


class I2a:
    def __init__(self, config):
        self.config = config
        with tf.name_scope('i2a_input'):
            self.input = tf.placeholder(tf.float32, self.config.state_size,
                                    name='states')
        self.initial_lstm_state = tf.placeholder(tf.float32, [2, None, self.config.lstm_size],
                                                 name='lstm_initial_state')
    def build_model(self):
        rollout=Rollout(self.input,self.initial_lstm_state,self.config)
        rollout_encoder=RolloutsEncoder(rollout.imagined_observations,rollout.imagined_rewards,self.config)