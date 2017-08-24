class BasePolicy:
    def __init__(self, sess, X_input, reuse=False):
        self.initial_state = []  # not stateful
        self.sess = sess
        self.X_input = X_input
        self.reuse = reuse
        self.value = None
        self.action = None

    def step(self, observation):
        raise NotImplementedError("step function not implemented")

    def value(self, observation):
        raise NotImplementedError("value function not implemented")
