import tensorflow as tf


class BaseTrainer:
    def __init__(self, sess, model, data, config, FLAGS):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data
        self.FLAGS = FLAGS
        self.cur_epoch_tensor = None
        self.cur_epoch_input = None
        self.cur_epoch_assign_op = None
        self.global_step_tensor = None
        self.global_step_input = None
        self.global_step_assign_op = None
        self.global_time_step_tensor = None
        self.global_time_step_input = None
        self.global_time_step_assign_op = None

        self.summary_placeholders = {}
        self.summary_ops = {}
        self.scalar_summary_tags = self.config.scalar_summary_tags
        # self.image_summary_tags = self.config.image_summary_tags

        # init the global step, global time step, the current epoch and the summaries
        self.init_global_step()
        self.init_global_time_step()
        self.init_cur_epoch()
        self.init_summaries()
        # self.init_image_summary()  This is a specific for a certain model.
        # To initialize all variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.summary_writer = tf.summary.FileWriter(self.config.summary_dir, self.sess.graph)

        if self.config.load or (not self.config.is_train):
            self.load()

    def save(self):
        print("Saving model...")
        self.saver.save(self.sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded")

    def init_cur_epoch(self):
        """
        Create cur epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

    def init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def init_global_time_step(self):
        """
        Create a global step variable to be a reference to the number of time steps
        :return:
        """
        with tf.variable_scope('global_time_step'):
            self.global_time_step_tensor = tf.Variable(0, trainable=False, name='global_time_step')
            self.global_time_step_input = tf.placeholder('int32', None, name='global_time_step_input')
            self.global_time_step_assign_op = self.global_step_tensor.assign(self.global_time_step_input)

    # summaries init
    def init_summaries(self):
        """
        Create the summary part of the graph
        :return:
        """
        with tf.variable_scope('train-summary'):
            for tag in self.scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

    def add_summary(self, step, summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step:
        :param summaries_dict:
        :param summaries_merged:
        :return:
        """
        if summaries_dict is not None:
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)
            self.summary_writer.flush()

    def init_image_summary(self):
        with tf.variable_scope('train-images-summary'):
            for tag in self.image_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32',
                                                                [None] + self.config.summary_image_shape,
                                                                name=tag)
                self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag],
                                                         max_outputs=self.config.max_images_to_visualize)
