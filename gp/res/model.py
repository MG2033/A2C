from gp.layers.action_conditional_lstm import actionlstm_cell
import tensorflow as tf


class ModelNetwork:
    def __init__(self, config):
        self.config = config
        self.summaries = None
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.initial_lstm_state = tf.placeholder(tf.float32, [2, None, 512], name='lstm_initial_state')

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, self.config.truncated_time_steps, self.config.state_size],
                                    name='X')
            self.y = tf.placeholder(tf.float32, [None, self.config.truncated_time_steps, self.config.state_size],
                                    name='y')
            self.rewards = tf.placeholder(tf.float32, [None, self.config.truncated_time_steps, 1],
                                          name='rewards')
            self.actions = tf.placeholder(tf.float32, [None, self.config.truncated_time_steps, self.config.action_dim],
                                          name='actions')

        self.build_model()

    def template(self, x, action, lstm_state):
        with tf.name_scope('hidden_layer_1'):
            h1 = tf.layers.dense(x, 128, kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn1 = tf.layers.batch_normalization(h1, training=self.is_training)
            drp1 = tf.layers.dropout(tf.nn.relu(bn1), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('hidden_layer_2'):
            h2 = tf.layers.dense(drp1, 256, kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn2 = tf.layers.batch_normalization(h2, training=self.is_training)
            drp2 = tf.layers.dropout(tf.nn.relu(bn2), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('hidden_layer_3'):
            h3 = tf.layers.dense(drp2, 256, kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn3 = tf.layers.batch_normalization(h3, training=self.is_training)
            drp3 = tf.layers.dropout(tf.nn.relu(bn3), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('lstm_layer') as scope:
            lstm_out, lstm_new_state = actionlstm_cell( drp3, lstm_state, action, 512, [None, 256],
                                                       self.config.action_dim,
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       activation=tf.tanh,scope=scope)

        with tf.name_scope('hidden_layer_4'):
            h4 = tf.layers.dense(lstm_out, 256, kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn4 = tf.layers.batch_normalization(h4, training=self.is_training)
            drp4 = tf.layers.dropout(tf.nn.relu(bn4), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('hidden_layer_5_1'):
            h5 = tf.layers.dense(drp4, 128, kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn5 = tf.layers.batch_normalization(h5, training=self.is_training)
            drp5_1 = tf.layers.dropout(tf.nn.relu(bn5), rate=self.config.dropout_rate, training=self.is_training,
                                       name='dropout')

        with tf.name_scope('next_state_output_layer'):
            next_state_out = tf.layers.dense(drp5_1, self.config.state_size, activation=None
                                             , kernel_initializer=tf.contrib.layers.xavier_initializer())

        if self.config.predict_reward:
            with tf.name_scope('hidden_layer_5_2'):
                h5_2 = tf.layers.dense(drp4, 128, kernel_initializer=tf.contrib.layers.xavier_initializer())
                drp5_2 = tf.layers.dropout(tf.nn.relu(h5_2), rate=self.config.dropout_rate, training=self.is_training,
                                           name='dropout')

            with tf.name_scope('reward_output_layer'):
                reward_out = tf.layers.dense(drp5_2, 1, activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        else:
            reward_out = None

        return next_state_out, reward_out, lstm_new_state

    def build_model(self):
        net_unwrap = []
        network_template = tf.make_template('network', self.template)

        lstm_state = self.initial_lstm_state
        lstm_state = tf.contrib.rnn.LSTMStateTuple(lstm_state[0], lstm_state[1])
        # lstm_state = None
        for i in range(self.config.truncated_time_steps):
            next_state_out, reward_out, lstm_state = network_template(self.x[:, i, :], self.actions[:, i], lstm_state)

            if self.config.predict_reward:
                out = tf.concat([next_state_out, reward_out], 1)
                net_unwrap.append(out)
            else:
                net_unwrap.append(next_state_out)

            if i == 0:
                self.first_step_out = (next_state_out, reward_out)
                self.first_step_lstm_state = lstm_state

        self.final_lstm_state = lstm_state

        with tf.name_scope('wrap_out'):
            net_unwrap = tf.stack(net_unwrap)
            self.output = tf.transpose(net_unwrap, [1, 0, 2])
            # self.output = tf.reshape(net_unwrap, [-1, self.config.truncated_time_steps, self.config.state_size])

        if self.config.predict_reward:
            labels = tf.concat([self.y, self.rewards], 2)
        else:
            labels = self.y

        self.loss = tf.losses.mean_squared_error(labels, self.output)



        # for batchnorm layers
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)
