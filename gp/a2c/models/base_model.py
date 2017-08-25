import tensorflow as tf


class BaseModel:
    def __init__(self):
        self.scalar_summary_tags = []
        self.scalar_summary_tags.extend(['policy-loss', 'policy-entropy', 'value-function-loss', 'reward'])
        self.merged_summaries = None
        self.global_step_tensor = None
        self.global_step_input = None
        self.global_step_assign_op = None
        self.init_global_step()

    def init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype='int32')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def merge_summaries(self):
        self.merged_summaries = tf.summary.merge_all()

    def init_input(self):
        raise NotImplementedError("init_input is not implemented")

    def init_network(self):
        raise NotImplementedError("init_network is not implemented")
