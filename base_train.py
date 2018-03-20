import tensorflow as tf


class BaseTrainer:
    def __init__(self, sess, model, args):
        self.model = model
        self.args = args
        self.sess = sess

        self.summary_placeholders = {}
        self.summary_ops = {}

    def save(self):
        print("Saving model...")
        self.saver.save(self.sess, self.args.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    def _load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Checkpoint loaded\n\n")
        else:
            print("No checkpoints available!\n\n")

    def __init_global_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)

    def _init_model(self):
        # init the global step, global time step, the current epoch and the summaries
        self.__init_global_step()
        self.__init_global_time_step()
        self.__init_cur_epoch()
        self.__init_global_saver()
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def __init_cur_epoch(self):
        """
        Create cur epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

    def __init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def __init_global_time_step(self):
        """
        Create a global time step variable to be a reference to the number of time steps
        :return:
        """
        with tf.variable_scope('global_time_step'):
            self.global_time_step_tensor = tf.Variable(0, trainable=False, name='global_time_step')
            self.global_time_step_input = tf.placeholder('int32', None, name='global_time_step_input')
            self.global_time_step_assign_op = self.global_time_step_tensor.assign(self.global_time_step_input)
