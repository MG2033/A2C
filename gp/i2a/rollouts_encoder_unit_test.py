import numpy as np
import tensorflow as tf

from gp.configs.i2a_config import I2AConfig
from gp.i2a.rollouts_encoder import RolloutsEncoder

obs_placeholder = tf.placeholder(tf.float32, [4, 5, 15, 19, 3])
rewards_placeholder = tf.placeholder(tf.float32, [4, 5, 1])

obs_arr = np.random.randint(0, 50, (4, 5, 15, 19, 3))
r_arr = np.random.randint(0, 50, (4, 5, 1))

config = I2AConfig()

encoder = RolloutsEncoder(obs_placeholder, rewards_placeholder, config)

output = encoder.rollout_encoding

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

out = sess.run(output, {obs_placeholder: obs_arr, rewards_placeholder: r_arr})

print(out.shape)