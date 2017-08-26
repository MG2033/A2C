from gp.res.res import Res
import tensorflow as tf



init = tf.global_variables_initializer()
config = tf.ConfigProto(
    device_count={'cpu': 0}
)
sess = tf.Session(config=config)
sess.run(init)
res = Res(sess)

res.train()
