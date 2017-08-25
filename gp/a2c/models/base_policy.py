class BasePolicy:
    def __init__(self, sess, input_shape, reuse=False):
        self.initial_state = []  # not stateful
        self.sess = sess
        self.input_shape = input_shape
        self.X_input = None
        self.reuse = reuse
        self.value_s = None
        self.action_s = None

    def step(self, observation):
        raise NotImplementedError("step function not implemented")

    def value(self, observation):
        raise NotImplementedError("value function not implemented")
