from gp.configs.res_config import ResConfig
from gp.res.model import RESModel
from gp.res.data_generator import GenerateData
from gp.res.train import Trainer


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
