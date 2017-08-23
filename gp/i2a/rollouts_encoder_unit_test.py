from gp.i2a.rollouts_encoder import RolloutsEncoder
from gp.i2a.i2a_config import I2AConfig
import tensorflow as tf
import numpy as np

input_placeholder = tf.placeholder(tf.float32, [5, 15, 19, 3])

input_arr = np.random.randint(0, 50, (5, 15, 19, 3))

config = I2AConfig()

encoder = RolloutsEncoder(None, None, config)

output = encoder.cnn_encoder(input_placeholder)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

out = sess.run(output, {input_placeholder: input_arr})

print(out.shape)