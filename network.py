import tensorflow as tf
import tensorflow.contrib.slim as slim


class Generator:
    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size

    def generate(self, noise):
        with tf.variable_scope(self.name):
            with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
                net = slim.fully_connected(noise, 2048)
                net = slim.fully_connected(net, self.batch_size * 8 * 8 * 128)
                net = tf.reshape(net, [-1, 8, 8, 128])
                net = slim.conv2d_transpose(net, 64, 2, 2, padding='VALID')
                net = slim.conv2d_transpose(net, 32, 2, 2, padding='VALID')
                net = slim.conv2d_transpose(net, 3, 2, 2, activation_fn=tf.nn.tanh, normalizer_fn=None, padding='VALID')
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
                            normalizer_fn=slim.batch_norm, reuse=tf.AUTO_REUSE, scope=self.name):
                net = slim.conv2d(image, 32, 5, 2, normalizer_fn=None, scope='d1')
                net = slim.conv2d(net, 64, 5, 2, scope='d2')
                net = slim.conv2d(net, 128, 5, 2, scope='d3')
                net = slim.conv2d(net, 256, 5, 2, scope='d4')
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1024, scope='f1')
                net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='f2')
                return net
