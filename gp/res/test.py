import tensorflow as tf
import numpy as np


class Test:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None] + [64], name='states')

        self.y = tf.placeholder(tf.float32, [None] + [64],
                                name='y')
        self.x_test = tf.placeholder(tf.float32, [None] + [64],
                                     name='states_test')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.model()

    def templete(self, x):
        with tf.name_scope('test_network'):
            h1 = tf.layers.dense(x, 64)
            bn1 = tf.layers.batch_normalization(h1, training=self.is_training)
            drp1 = tf.layers.dropout(tf.nn.relu(bn1), rate=.9, training=self.is_training,
                                     name='dropout')
            h2 = tf.layers.dense(drp1, 64)
            bn2 = tf.layers.batch_normalization(h2, training=self.is_training)
            out = tf.layers.dropout(tf.nn.relu(bn2), rate=.9, training=self.is_training,
                                    name='dropout')

            return out

    def model(self):
        self.network_template = tf.make_template('network', self.templete)
        self.out = self.x
        for i in range(5):
            self.out = self.network_template(self.out)


        self.loss = tf.losses.mean_squared_error(self.out, self.y)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_step = tf.train.RMSPropOptimizer(.00002).minimize(self.loss)

        self.out_test = self.network_template(self.x_test)





def main(_):
    my_test = Test()

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    batch_x = np.zeros((4, 64))
    batch_y = np.zeros((4, 64))
    for i in range(10):
        feed_dict = {my_test.x: batch_x, my_test.y: batch_y
            , my_test.is_training: True}
        _, loss = sess.run(
            [my_test.train_step, my_test.loss]
            , feed_dict)


if __name__ == '__main__':
    tf.app.run()

