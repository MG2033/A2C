import tensorflow as tf


class EnvSummaryLogger:
    """
    Helper class to summarize all environments at the same time on the same plots.
    """

    def __init__(self, sess, summary_dirs):
        self.sess = sess
        self.summary_writer = [tf.summary.FileWriter(summary_dirs[i], self.sess.graph)
                               for i in range(len(summary_dirs))]
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.env_summary_tags = ['reward', 'episode_length']
        self.init_summaries()

    def init_summaries(self):
        """
        Create the summary part of the graph
        :return:
        """
        with tf.variable_scope('env-train-summaries'):
            for tag in self.env_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

    def add_summary_all(self, step, summaries_arr_dict=None, summaries_merged=None):
        for i in range(len(summaries_arr_dict)):
            if summaries_arr_dict[i]['reward'] != -1:
                self.add_summary(i, step, summaries_arr_dict[i], summaries_merged)

    def add_summary(self, id, step, summaries_dict=None, summaries_merged=None):
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
                self.summary_writer[id].add_summary(summary, step)
            self.summary_writer[id].flush()
        if summaries_merged is not None:
            self.summary_writer[id].add_summary(summaries_merged, step)
            self.summary_writer[id].flush()
