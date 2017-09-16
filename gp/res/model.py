from gp.layers.action_conditional_lstm import actionlstm_cell
import tensorflow as tf


class RESModel:
    def __init__(self, config):
        """
        :param config: configration object
        """
        self.config = config
        self.summaries = None
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.name_scope('train_inputs'):
            self.initial_lstm_state = tf.placeholder(tf.float32, [2, None, self.config.lstm_size],
                                                     name='lstm_initial_state')
            self.x = tf.placeholder(tf.float32, [None, self.config.truncated_time_steps] + self.config.state_size,
                                    name='states')
            self.y = tf.placeholder(tf.int32, [None, self.config.truncated_time_steps] + self.config.labels_size,
                                    name='next_states')
            self.rewards = tf.placeholder(tf.float32, [None, self.config.truncated_time_steps, 1],
                                          name='rewards')
            self.actions = tf.placeholder(tf.float32, [None, self.config.truncated_time_steps, self.config.action_dim],
                                          name='actions')
        with tf.name_scope('test_inputs'):
            self.x_test = tf.placeholder(tf.float32, [None] + self.config.state_size,
                                         name='states_test')
            self.initial_lstm_state_test = tf.placeholder(tf.float32, [2, None, self.config.lstm_size],
                                                          name='lstm_state_test')
            self.actions_test = tf.placeholder(tf.float32, [None, self.config.action_dim],
                                               name='actions_test')

    def template(self, x, action, lstm_state):

        """
        :param x: input tensor of shape: [None, truncated_time_steps ] + self.config.state_size
        :param action: input tensor of shape:[None, truncated_time_steps, action_dim]
        :param lstm_state: input tensor of shape: [2, lstm_size, lstm_size]
        :return: the output and the lstm hidden state
        """

        with tf.name_scope('encoder_1'):
            h1 = tf.layers.conv2d(x, 64, kernel_size=(8, 8), strides=(2, 2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            bn1 = tf.layers.batch_normalization(h1, training=self.is_training)
            drp1 = tf.layers.dropout(tf.nn.relu(bn1), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('encoder_2'):
            h2 = tf.layers.conv2d(drp1, 32, kernel_size=(6, 6), strides=(2, 2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            bn2 = tf.layers.batch_normalization(h2, training=self.is_training)
            drp2 = tf.layers.dropout(tf.nn.relu(bn2), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('encoder_3'):
            h3 = tf.layers.conv2d(drp2, 32, kernel_size=(6, 6), strides=(2, 2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            bn3 = tf.layers.batch_normalization(h3, training=self.is_training)
            drp3 = tf.layers.dropout(tf.nn.relu(bn3), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('encoder_4'):
            h4 = tf.layers.conv2d(drp3, 32, kernel_size=(4, 4), strides=(2, 2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            bn4 = tf.layers.batch_normalization(h4, training=self.is_training)
            drp4 = tf.layers.dropout(tf.nn.relu(bn4), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('flatten_1'):
            encoded = tf.contrib.layers.flatten(drp4)

        # the size of encodded vector
        encoded_vector_size = encoded.get_shape()[1]

        with tf.name_scope('lstm_layer') as scope:
            lstm_out, lstm_new_state = actionlstm_cell(encoded, lstm_state, action, self.config.lstm_size,
                                                       self.config.action_dim,
                                                       initializer=tf.contrib.layers.xavier_initializer(),
                                                       activation=tf.tanh, scope='lstm_layer')

        with tf.name_scope('hidden_layer_1'):
            h5 = tf.layers.dense(lstm_out, encoded_vector_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn5 = tf.layers.batch_normalization(h5, training=self.is_training)
            drp5 = tf.layers.dropout(tf.nn.relu(bn5), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('reshape_1'):
            # the last encoder conv layer shape
            deconv_init_shape = drp4.get_shape().as_list()
            reshaped_drp4 = tf.reshape(drp5, [-1] + deconv_init_shape[1:])

        with tf.name_scope('decoder_1'):
            h6 = tf.layers.conv2d_transpose(reshaped_drp4, 32, kernel_size=(4, 4), strides=(2, 2), padding='SAME',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn6 = tf.layers.batch_normalization(h6, training=self.is_training)
            drp6 = tf.layers.dropout(tf.nn.relu(bn6), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('decoder_2'):
            h7 = tf.layers.conv2d_transpose(drp6, 32, kernel_size=(6, 6), strides=(2, 2),
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            bn7 = tf.layers.batch_normalization(h7, training=self.is_training)
            drp7 = tf.layers.dropout(tf.nn.relu(bn7), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('decoder_3'):
            h8 = tf.layers.conv2d_transpose(drp7, 32, kernel_size=(6, 6), strides=(2, 2),
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            bn8 = tf.layers.batch_normalization(h8, training=self.is_training)
            drp8 = tf.layers.dropout(tf.nn.relu(bn8), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('decoder_4'):
            h9 = tf.layers.conv2d_transpose(drp8, 64, kernel_size=(8, 8), strides=(2, 2), padding='SAME',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn9 = tf.layers.batch_normalization(h9, training=self.is_training)
            drp9 = tf.layers.dropout(tf.nn.relu(bn9), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('decoder_5'):
            next_state_out = tf.layers.conv2d(drp9, 2, kernel_size=(3, 3), strides=(1, 1),
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            next_state_out_softmax = tf.nn.softmax(next_state_out)

        if self.config.predict_reward:
            with tf.name_scope('reward_flatten'):
                flattened_drp7 = tf.contrib.layers.flatten(drp7)

            with tf.name_scope('reward_hidden_layer_2'):
                h7_2 = tf.layers.dense(flattened_drp7, 128, kernel_initializer=tf.contrib.layers.xavier_initializer())
                drp7_2 = tf.layers.dropout(tf.nn.relu(h7_2), rate=self.config.dropout_rate, training=self.is_training,
                                           name='dropout')

            with tf.name_scope('reward_output_layer'):
                reward_out = tf.layers.dense(drp7_2, 1, activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        else:
            reward_out = None

        # print encoder_decoder layers shape for debugging
        # print(drp1.get_shape().as_list())
        # print(drp2.get_shape().as_list())
        # print(drp3.get_shape().as_list())
        # print(drp4.get_shape().as_list())
        # print(drp6.get_shape().as_list())
        # print(drp7.get_shape().as_list())
        # print(drp8.get_shape().as_list())
        # print(next_state_out.get_shape().as_list())

        return next_state_out, next_state_out_softmax, reward_out, lstm_new_state

    def build_model(self):
        net_unwrap = []
        net_softmax_unwrap = []

        reward_unwrap = []
        self.network_template = tf.make_template('network', self.template)

        lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state[0], self.initial_lstm_state[1])
        for i in range(self.config.truncated_time_steps):
            if i >= self.config.observation_steps_length:
                state_out, next_state_out_softmax, reward_out, lstm_state = self.network_template(
                    next_state_out_softmax,
                    self.actions[:, i],
                    lstm_state)
            else:
                state_out, next_state_out_softmax, reward_out, lstm_state = self.network_template(self.x[:, i, :],
                                                                                                  self.actions[:, i],
                                                                                                  lstm_state)

            if self.config.predict_reward:
                reward_unwrap.append(reward_out)
                net_unwrap.append(state_out)
                net_softmax_unwrap.append(next_state_out_softmax)
            else:
                net_unwrap.append(state_out)
                net_softmax_unwrap.append(next_state_out_softmax)


        self.final_lstm_state = lstm_state
        with tf.name_scope('wrap_out'):
            net_unwrap = tf.stack(net_unwrap)
            self.output = tf.transpose(net_unwrap, [1, 0, 2, 3, 4])

            net_softmax_unwrap = tf.stack(net_softmax_unwrap)
            self.output_softmax = tf.transpose(net_softmax_unwrap, [1, 0, 2, 3, 4])

            if self.config.predict_reward:
                reward_unwrap = tf.stack(reward_unwrap)
                self.reward_output = tf.stack(reward_unwrap)
                self.reward_output = tf.transpose(self.reward_output, [1, 0, 2])

        with tf.name_scope('loss'):
            # state loss
            self.states_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
            self.loss = self.states_loss
            # adding reward loss
            if self.config.predict_reward:
                self.reward_loss = tf.losses.mean_squared_error(self.reward_output, self.rewards)
                self.loss += self.reward_loss

        with tf.name_scope('train_step'):
            # for batchnorm layers
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                # RMSProp as in paper
                self.train_step = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(self.loss)

        # test_model
        lstm_state_test = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state_test[0],
                                                        self.initial_lstm_state_test[1])
        self.output_test, self.output_softmax_test, self.reward_out_test, self.lstm_state_test = self.network_template(
            self.x_test,
            self.actions_test,
            lstm_state_test)
