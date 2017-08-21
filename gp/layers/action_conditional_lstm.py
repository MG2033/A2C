import tensorflow as tf


class ActionLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, num_units, w_hv, w_hz, w_vh, w_va,
                 activation=None, reuse=None):
        super(tf.nn.rnn_cell.BasicLSTMCell, self).__init__(_reuse=reuse)

        self._num_units = num_units
        self._activation = activation

        # variables of action conditional lstm equations
        self._w_hv = w_hv
        self._w_hz = w_hz
        self._w_vh = w_vh
        self._w_va = w_va

    def __call__(self, x, h, a, scope=None):
        with tf.variable_scope(self.scope or self.__class__.__name__):
            previous_memory, previous_output = h

            v = tf.matmul(self._w_vh, tf.transpose(previous_output, (1, 0))) * tf.matmul(self._w_va,
                                                                                         tf.transpose(a, (1, 0)))
            w_v = tf.matmul(self._w_hv, v)
            iv, fv, ov, cv = tf.split(w_v, 4, axis=0)
            w_z = tf.matmul(self._w_hz, tf.transpose(x))
            iz, fz, oz, cz = tf.split(w_z, 4, axis=0)

            i = tf.sigmoid(iv + iz)
            f = tf.sigmoid(fv + fz)
            o = tf.sigmoid(ov + oz)
            memory = f * tf.transpose(previous_memory, (1, 0)) + i * tf.tanh(cv + cz)
            output = o * tf.tanh(memory)

        return tf.transpose(output, (1, 0)), tf.contrib.rnn.LSTMStateTuple(tf.transpose(memory, (1, 0)),
                                                                           tf.transpose(output, (1, 0)))


def actionlstm_cell(x, h, a, num_units, input_shape, action_dim,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    activation=tf.tanh, scope='action_lstm'):
    with tf.variable_scope(scope) as scope_:
        # Initialize the weights
        state_size = input_shape[1]

        w_hv = tf.get_variable('w_hv', [4 * num_units, 2 * num_units], initializer=initializer)
        w_hz = tf.get_variable('w_hz', [4 * num_units, state_size], initializer=initializer)
        w_vh = tf.get_variable('w_vh', [2 * num_units, num_units], initializer=initializer)
        w_va = tf.get_variable('w_va', [2 * num_units, action_dim], initializer=initializer)

        # init the cell
        cell = ActionLSTMCell(num_units, w_hv, w_hz, w_vh, w_va, activation)
        # call the cell
        if h is None:
            h = cell.zero_state(tf.shape(x)[0], tf.float32)

        output, state = cell(x, h, a)

    return output, state
