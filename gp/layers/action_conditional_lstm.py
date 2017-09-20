import tensorflow as tf


class ActionLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, num_units, w_all_v, w_all_z, wh, wa,
                 activation=None, reuse=None, scope=None):
        """
        :param num_units: lstm num units
        :param w_all_v:
        :param w_all_z:
        :param wh:
        :param wa:
        :param activation: activation function
        :param reuse: whether to reuse variables or not
        :param scope: scope name
        """
        super(tf.nn.rnn_cell.BasicLSTMCell, self).__init__(_reuse=reuse)
        self.scope = scope
        self._num_units = num_units
        self._activation = activation

        # variables of action conditional lstm equations
        self._w_all_v = w_all_v
        self._w_all_z = w_all_z
        self._wh = wh
        self._wa = wa

    def __call__(self, x, h, a, scope=None):
        with tf.variable_scope(self._scope or self.__class__.__name__):
            previous_memory, previous_output = h

            v = tf.matmul(self._wh, tf.transpose(previous_output, (1, 0))) * tf.matmul(self._wa,
                                                                                       tf.transpose(a, (1, 0)))
            w_v = tf.matmul(self._w_all_v, v)
            iv, fv, ov, cv = tf.split(w_v, 4, axis=0)
            w_z = tf.matmul(self._w_all_z, tf.transpose(x))
            iz, fz, oz, cz = tf.split(w_z, 4, axis=0)

            i = tf.sigmoid(iv + iz)
            f = tf.sigmoid(fv + fz)
            o = tf.sigmoid(ov + oz)
            memory = f * tf.transpose(previous_memory, (1, 0)) + i * tf.tanh(cv + cz)
            output = o * tf.tanh(memory)

        return tf.transpose(output, (1, 0)), tf.contrib.rnn.LSTMStateTuple(tf.transpose(memory, (1, 0)),
                                                                           tf.transpose(output, (1, 0)))


def actionlstm_cell(x, h, a, num_units, action_dim,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    activation=tf.tanh, scope='action_lstm'):
    """
    :param x:  state input
    :param h: hidden state tuple input
    :param a: action input
    :param num_units: lstm num units
    :param action_dim: input action dimension
    :param initializer: initializer function
    :param activation: activation function
    :param scope: scope name
    :return: lstm output and state
    """
    with tf.variable_scope(scope) as scope_:
        # Initialize the weights
        state_size = x.get_shape()[1]

        w_all_v = tf.get_variable('w_all_v', [4 * num_units, 2 * num_units], initializer=initializer)
        w_all_z = tf.get_variable('w_all_z', [4 * num_units, state_size], initializer=initializer)
        wh = tf.get_variable('wh', [2 * num_units, num_units], initializer=initializer)
        wa = tf.get_variable('wa', [2 * num_units, action_dim], initializer=initializer)

    # init the cell
    cell = ActionLSTMCell(num_units, w_all_v, w_all_z, wh, wa, activation)
    # call the cell
    if h is None:
        h = cell.zero_state(tf.shape(x)[0], tf.float32)

    output, state = cell(x, h, a)

    return output, state