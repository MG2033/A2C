import tensorflow as tf
import numpy as np
import os

from gp.configs.i2a_config import I2AConfig
from gp.i2a.rollout_encoder.rollouts_encoder import RolloutsEncoder
from gp.logger.logger import Logger

tf.logging.set_verbosity(tf.logging.DEBUG)
tf.reset_default_graph()
sess = tf.Session()

summary_dir = '/tmp/summary'
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

obs_placeholder = tf.placeholder(tf.float32, [4, 5, 15, 19, 3])
rewards_placeholder = tf.placeholder(tf.float32, [4, 5, 1])

obs_arr = np.random.randint(0, 50, (4, 5, 15, 19, 3))
r_arr = np.random.randint(0, 50, (4, 5, 1))

config = I2AConfig()

Logger.info('started building the rollouts encoder')

with tf.name_scope('rollout_encoder'):
    encoder = RolloutsEncoder(obs_placeholder, rewards_placeholder, config)

Logger.info('finished building the rollouts encoder')

output, merged = encoder.rollout_encoding

init = tf.global_variables_initializer()

sess.run(init)

logger = Logger(sess, summary_dir)

out, summary = sess.run([output, merged], {obs_placeholder: obs_arr, rewards_placeholder: r_arr})

logger.add_merged_summary(3, summary)

logger.shape(output)

logger.close()