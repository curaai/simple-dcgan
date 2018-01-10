import tensorflow as tf
import tensorflow.contrib.slim as slim


class Generator:
    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size

    def generate(self, noise):
        with tf.variable_scope(self.name):
            with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                net = slim.fully_connected(noise, 4 * 4 * 1024 )
                net = tf.reshape(net, [-1, 4, 4, 1024])
                net = slim.conv2d_transpose(net, 512, [5, 5], [2, 2], padding='SAME')
                net = slim.conv2d_transpose(net, 256, [5, 5], [2, 2], padding='SAME')
                net = slim.conv2d_transpose(net, 128, [5, 5], [2, 2], padding='SAME')
                net = slim.conv2d_transpose(net, 3, [5, 5], [2, 2], padding='SAME',
                                            activation_fn=tf.nn.tanh)
                return net


class Discriminator:
    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size
        self.reuse = True

    # original size = (128, 128, 3)
    def discriminate(self, image):
        with tf.variable_scope(self.name):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm, reuse=tf.AUTO_REUSE, scope=self.name,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                net = slim.conv2d(image, 32, 4, 2, scope='d1')
                net = slim.conv2d(net, 64, 4, 2, scope='d2')
                net = slim.conv2d(net, 128, 4, 2, scope='d3')
                net = slim.conv2d(net, 256, 4, 2, scope='d4')
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='f1')
                return net
