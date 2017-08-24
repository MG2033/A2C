import tensorflow as tf
from gp.utils.utils import create_experiment_dirs


class BaseTrainer:
    def __init__(self, sess, model, max_to_keep, experiment_dir, is_train, cont_train):

        self.sess = sess
        # Choosing the model then build it
        self.model = model
        self.summary_dir, self.checkpoint_dir = create_experiment_dirs(experiment_dir)
        self.checkpoint_dir = self.checkpoint_dir
        self.build_model()
        # init the summaries
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.scalar_summary_tags = self.model.scalar_summary_tags
        self.init_summaries()

        # To initialize all variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.saver = tf.train.Saver(max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=1,
                                    save_relative_paths=True)
        self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        if cont_train or (not is_train):
            self.load()

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

    def save(self, verbose=0):
        """
        Save Model Checkpoint
        :return:
        """
        if verbose == 1:
            print("Saving the model..")
        self.saver.save(self.sess, self.checkpoint_dir, self.model.global_step_tensor)
        if verbose == 1:
            print("Model saved successfully")

    def load(self):
        """
        Load the latest checkpoint
        :return:
        """
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Model loaded from the latest checkpoint successfully")
        else:
            print("\nFirst time to train..\n")

    def build_model(self):
        """
        This will contain the building logic of any model and can be written in it all the model
        Do what you want to do
        :return:
        """
        raise NotImplementedError("build_model function is not implemented")

    def train(self):
        raise NotImplementedError("train function is not implemented")

    def test(self):
        raise NotImplementedError("test function is not implemented")
