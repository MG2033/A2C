import tensorflow as tf
from gp.i2a.imagination_core.rollout_policy import RolloutPolicy
from gp.res.model import RESModel
import numpy as np
from gp.configs.res_config import ResConfig

class Rollout:
    def __init__(self, observation, lstm_state, config):
        self.config = config
        self.observation = observation
        res_config=ResConfig()
        res_model = RESModel(res_config)
        self.res_templete = res_model.template()
        self.policy_templete = RolloutPolicy.policy_template('rollout_policy', config.actions_num)
        self.initial_lstm_state = lstm_state
        self.next_lstm_states = []
        self.build_rollout_model()

    def build_rollout_model(self):
        rollout_policy_network = tf.make_template('rollout_policy_network', self.policy_templete)
        rollout_res_network = tf.make_template('rollout_res_network', self.res_templete)

        observations_unwrap = []
        rewards_unwrap = []

        # the actual prediction of the rollout policy (for loss)
        self.rollout_actions_prob = rollout_policy_network(self.observation)

        actions = tf.constant(np.eye(self.config.actions_num), dtype=tf.float32,
                              shape=[self.config.actions_num] * 2)
        multiples = tf.constant([self.config.actions_num, 1, 1, 1], tf.int32, shape=[4])
        observations = tf.tile(self.observation, multiples)
        lstm_states = self.initial_lstm_state
        for i in range(self.config.rollouts_steps):
            observations, rewards, lstm_states = rollout_res_network(observations, actions,
                                                                     lstm_states)
            observations_unwrap.append(observations)
            rewards_unwrap.append(rewards)
            actions_prob = rollout_policy_network(observations)
            actions = tf.arg_max(actions_prob,1)

        observations_unwrap = tf.stack(observations_unwrap)
        rewards_unwrap = tf.stack(rewards_unwrap)

        self.imagined_observations = tf.transpose(observations_unwrap, [1, 0, 2, 3, 4])
        self.imagined_rewards = tf.transpose(rewards_unwrap, [1, 0, 2])
        # return self.imagined_observations,self.imagined_rewards,self.rollout_actions_prob