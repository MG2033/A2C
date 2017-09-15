import tensorflow as tf


# source: https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow
class ActionConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """

    def __call__(self, inputs, state, actions, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self.shape
        num_features = self.num_features
        zeros = tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, shape[0], shape[1], num_features]),
                                              tf.zeros([batch_size, shape[0], shape[1], num_features]))
        return zeros


class BasicActionConvLSTMCell(ActionConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, activation=tf.nn.tanh, initializer=None):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)
        # if input_size is not None:
        #     logging.warn("%s: The input_size parameter is deprecated.", self)

        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._initializer = initializer

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, actions, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=3, num_or_size_splits=2, value=state)

            # ##########################################################
            # a: [None, w, h, actions_num]
            # h: [None, w, h, lstm_filters]
            # c: [None, w, h, lstm_filters]
            # inputs: [None, w, h, c]
            #
            # w_h: [3, 3, lstm_filters, lstm_filters]
            # w_a: [3, 3, actions_num, lstm_filters]
            # w_inputs: [3, 3, c, lstm_filters]
            # w_v: [3, 3, lstm_filters, lstm_filters]
            # w_c_v: [3, 3, lstm_filters, lstm_filters]
            # w_c_inputs: [3, 3, c, lstm_filters]
            #
            # v = w_h * h . w_a * a
            # i_1, f_1, o_1 = inputs * w_inputs
            # i_2, f_2, o_2 = v * w_v
            #
            # i = sigmoid(i_1 + i_2)
            # f = sigmoid(f_1 + f_2)
            # o = sigmoid(o_1 + o_2)
            #
            # c_ = f . c + i . tanh(w_c_v * v + w_c_input * inputs)
            # h_ = o . tanh(c_)
            # ##########################################################

            v = _conv_linear([h], self.filter_size, self.num_features, True, scope=scope + 'w_h', initializer=self._initializer) + \
                _conv_linear([actions], self.filter_size, self.num_features, True, scope=scope + 'w_a', initializer=self._initializer)

            concat_1 = _conv_linear([inputs], self.filter_size, 3 * self.num_features, True, scope=scope + 'w_inputs', initializer=self._initializer)
            concat_2 = _conv_linear([v], self.filter_size, 3 * self.num_features, True, scope=scope + 'w_v', initializer=self._initializer)
            i_1, f_1, o_1 = tf.split(axis=3, num_or_size_splits=3, value=concat_1)
            i_2, f_2, o_2 = tf.split(axis=3, num_or_size_splits=3, value=concat_2)

            i = tf.nn.sigmoid(i_1 + i_2)
            f = tf.nn.sigmoid(f_1 + f_2)
            o = tf.nn.sigmoid(o_1 + o_2)

            c_1 = _conv_linear([inputs], self.filter_size, self.num_features, True, scope=scope + 'w_c_inputs', initializer=self._initializer)
            c_2 = _conv_linear([v], self.filter_size, self.num_features, True, scope=scope + 'w_c_v', initializer=self._initializer)

            new_c = f * c + i * tf.tanh(c_1 + c_2)
            new_h = o * tf.tanh(new_c)

            if self._state_is_tuple:
                new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(axis=3, values=[new_c, new_h])
            return new_h, new_state


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None, initializer=None):
    """convolution:
    Args:
      args: a 4D Tensor or a list of 4D, batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      num_features: int, number of features.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 4D Tensor with shape [batch h w num_features]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope):
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype, initializer=initializer)
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term
