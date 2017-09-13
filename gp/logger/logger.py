import tensorflow as tf


class Logger:
    def __init__(self, sess, log_dir):
        """
        :param log_dir: logging directory for tensorboard
        :param verbosity_level: the level of logging to be printed
        """
        self.sess = sess
        self.summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        self.summary_inputs = {}
        self.summary_ops = {}

    def initialize_scalar_summary(self, tag):
        self.summary_inputs[tag] = tf.placeholder(tf.float32, name=tag)
        self.summary_ops[tag] = tf.summary.scalar(tensor=self.summary_inputs[tag], name=tag)

    def initialize_image_summary(self, tag, shape):
        self.summary_inputs[tag] = tf.placeholder(tf.float32, [None] + shape, name=tag)
        self.summary_ops[tag] = tf.summary.image(tensor=self.summary_inputs[tag], name=tag)

    def add_merged_summary(self, step, merged_summary):
        self.summary_writer.add_summary(merged_summary, step)

    def add_scalar_summary(self, step, summaries_dict):
        ops_list = []
        feed_dict = {}
        for tag, value in summaries_dict.items():
            if tag not in self.summary_ops:
                self.initialize_scalar_summary(tag)
            ops_list.append(self.summary_ops[tag])
            feed_dict[self.summary_inputs[tag]] = value

        merged = tf.summary.merge(ops_list)

        summary = self.sess.run(merged, feed_dict)
        self.summary_writer.add_summary(summary, step)

    def add_image_summary(self, step, summaries_dict):
        ops_list = []
        feed_dict = {}
        for tag, value in summaries_dict.items():
            if tag not in self.summary_ops:
                self.initialize_image_summary(tag, value.shape[1:])
            ops_list.append(self.summary_ops[tag])
            feed_dict[self.summary_inputs[tag]] = value

        merged = tf.summary.merge(ops_list)

        summary = self.sess.run(merged, feed_dict)
        self.summary_writer.add_summary(summary, step)

    def add_run_metadate_summary(self, step, run_metadata):
        self.summary_writer.add_run_metadata(run_metadata, 'step:%d' % step)

    @staticmethod
    def summarize_scalar(scalar_tensor, name, collection):
        tf.summary.scalar(name, scalar_tensor, collections=[collection])

    @staticmethod
    def summarize_images(images_tensor, name, collection, images_num=10):
        tf.summary.image(name, images_tensor, images_num, collections=[collection])

    @staticmethod
    def summarize_histogram(histogram_tensor, name, collection):
        tf.summary.histogram(name, histogram_tensor, collections=[collection])

    @staticmethod
    def summarize_layer(tensor, collection):
        tf.add_to_collection(collection, tf.contrib.layers.summarize_activation(tensor))

    @staticmethod
    def info(msg):
        tf.logging.info(msg)

    @staticmethod
    def warn(msg):
        tf.logging.warn(msg)

    @staticmethod
    def error(msg):
        tf.logging.error(msg)

    @staticmethod
    def shape(tensor):
        tf.logging.debug(tensor.get_shape().as_list())

    def close(self):
        self.summary_writer.close()
