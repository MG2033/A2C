from gp.configs.res_config import ResConfig
from gp.res.model import RESModel
from gp.res.data_generator import GenerateData
from gp.res.train import Trainer
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('cont_train', False , """ whether to Load the Model and Continue Training or not """)

class Res:
    def __init__(self, sess):
        """
        :param sess: the tensorflow session
        """
        self.sess = sess
        self.config = ResConfig()
        self.model = RESModel(self.config)
        self.model.build_training_model()
        self.data = GenerateData(self.config)
        self.trainer = Trainer(self.sess, self.model, self.data, self.config)

    def train(self):
        self.trainer.train()





def main(_):
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    res = Res(sess)

    res.train()

    if FLAGS.is_train:
        res.train()
if __name__ == '__main__':
    tf.app.run()