import tensorflow as tf
import numpy as np

from PIL import Image
from network import Discriminator, Generator
from cats import get_from_dump, dump_image
images = get_from_dump('dump.pkl')


def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))


def comon_nyan(batch_size):
    rand_mask = np.random.choice(1000, batch_size, replace=False)
    return images[rand_mask]


if __name__ == '__main_':
    dump_image('new_cat', 'dump.pkl')

if __name__ == '__main__':
    total_epoch = 200
    batch_size = 20
    learning_rate = 0.0002

    total_batch = int(1000 / batch_size)

    # about training image, (128, 128, 3) 이미지
    n_length = 64
    n_channel = 3

    # noise vector dimension
    n_noise = 200

    G = Generator('G', batch_size)
    D = Discriminator('D', batch_size)

    data_x = tf.placeholder(tf.float32, [None, n_length, n_length, n_channel])
    z = tf.placeholder(tf.float32, [None, n_noise])

    data_g = G.generate(z)
    data_g = tf.reshape(data_g, [-1, n_length, n_length, n_channel])

    real_D = D.discriminate(data_x)
    fake_D = D.discriminate(data_g)

    # loss_D = -tf.reduce_mean(tf.log(real_D) + tf.log(1-fake_D))
    # loss_G = -tf.reduce_mean(tf.log(fake_D))
    loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D, labels=tf.ones_like(real_D))) \
             + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D, labels=tf.zeros_like(fake_D)))
    loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D, labels=tf.ones_like(fake_D)))

    varlist_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=D.name)
    varlist_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G.name)

    train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=varlist_D)
    train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=varlist_G)

    print("start")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(total_epoch):
            for i in range(total_batch):
                batch_cats = comon_nyan(batch_size)
                batch_x = np.reshape(batch_cats, (batch_size, n_length, n_length, n_channel))
                noise = get_noise(batch_size, n_noise)

                _, loss_var_D = sess.run([train_D, loss_D], feed_dict={data_x: batch_x, z: noise})
                _, loss_var_G = sess.run([train_G, loss_G], feed_dict={z: noise})

            print('Epoch:', '%04d' % epoch)

            if epoch % 5 == 0:
                sample_size = 10
                noise = get_noise(sample_size, n_noise)
                samples = sess.run(data_g, feed_dict={z: noise})

                for i in range(sample_size):
                    img = Image.fromarray(samples[i], 'RGB')
                    img.save('samples/' + str(epoch)+'_'+str(i)+'.png')
