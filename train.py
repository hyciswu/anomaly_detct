
from data import cifar10, mnist, brain
from model import allcnn, WGAN, classifier
import tensorgraph as tg
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import tensorflow.contrib as tc
from tensorflow.python.framework import ops

hvd.init()

# def train():
graph = tf.Graph()
with graph.as_default():
    batch_size = 32
    learning_rate = 0.0002
    patch_size = [20, 20]


    # logdir = './log/cifar10'
    # names_records, h, w, c, n_exp = cifar10(create_tfrecords=False, batch_size=batch_size)
    #
    # logdir = './log/mnist'
    names_records, h, w, c, n_exp = mnist(create_tfrecords=False, batch_size=batch_size)

    '''
    sess = tf.Session()
    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
    ttl = 0
    for i in range(10000):
        ttl += len(sess.run(names_records['X']))
        print(ttl)
    '''
    if hvd.rank() == 0:
        ts = tg.utils.ts()
        logdir = './log/brain/{}'.format(ts)
    names_records, h, w, c, n_exp = brain(create_tfrecords=False, batch_size=batch_size, shape=[200,200])
    # names_records['X']
    # names_records['X']
    # seq = allcnn(nclass=10, h=32, w=32, c=3)
    # g_loss, d_loss, X_sampler_g = wgan(X_real_sb=names_records['X'], height=h, width=w, channel=c, batch_size=batch_size)
    z_dim = 100
    z_ph = tf.placeholder('float32', [None, z_dim])
    with tf.variable_scope('GAN'):
        model = WGAN(h, w, c, z_dim=z_dim, gf_dim=64, df_dim=64)
        g_loss, d_loss, X_sampler_g, _ = model.loss(X_real_sb=names_records['X'],
                                                  z_ph=z_ph,
                                                  batch_size=batch_size,
                                                  improved_wgan=True, weight_regularize=True)



    # X_gen_patch_ph = tf.placeholder('float32', [None] + patch_size + [1])
    # X_patch_ph = tf.placeholder('float32', [None] + patch_size + [1])
    # h_p, w_p = patch_size
    # with tf.variable_scope('Classifier'):
    #     y_train_sb, y_test_sb = classifier(X_patch_ph, X_gen_patch_ph, h_p, w_p)

    # summary = tf.summary()
    X_sampler_g_sm = tf.summary.image('X_sampler_g', X_sampler_g, max_outputs=10)
    # y_train_sb = seq.train_fprop(names_records['X'])
    # y_test = seq.test_fprop()
    # loss_train_sb = tg.cost.mse(y_train_sb, names_records['y'])
    # accu_train_sb = tg.cost.accuracy(y_train_sb, names_records['y'])
#
    # import pdb; pdb.set_trace()
    # gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Generator')
    # dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Discriminator')
    # opt = tf.train.RMSPropOptimizer(learning_rate)
    # d_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9)
    # g_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9)

    # update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    # import pdb; pdb.set_trace()
    dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'GAN/Discriminator')
    d_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9)
    d_opt = hvd.DistributedOptimizer(d_opt)
    update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS, 'GAN/Discriminator')
    with ops.control_dependencies(update_ops):
        d_loss_op = d_opt.minimize(d_loss, var_list=dis_var_list)


    gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'GAN/Generator')
    g_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9)
    g_opt = hvd.DistributedOptimizer(g_opt)
    update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS, 'GAN/Generator')
    with ops.control_dependencies(update_ops):
        g_loss_op = g_opt.minimize(g_loss, var_list=gen_var_list)



#     # import pdb; pdb.set_trace()


    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in dis_var_list]

        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
        #         .minimize(self.d_loss, var_list=self.d_net.vars)
        #     self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
        #         .minimize(self.g_loss, var_list=self.g_net.vars)

    # g_opt = hvd.DistributedOptimizer(g_opt)
    # d_opt = hvd.DistributedOptimizer(d_opt)

    # train_op = opt.minimize(loss_train_sb)
    # g_loss_op = g_opt.minimize(g_loss, var_list=gen_var_list)
    # d_loss_op = d_opt.minimize(d_loss, var_list=dis_var_list)



    # init = tf.global_variables_initializer()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    bcast = hvd.broadcast_global_variables(0)

    # import pdb; pdb.set_trace()
#
# # Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
# import pdb; pdb.set_trace()



with tf.Session(graph=graph, config=config) as sess:
# with tf.Session(graph=graph) as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(init_op)
    bcast.run()



    # n_exp = 50000

    if hvd.rank() == 0:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'GAN')
        saver = tf.train.Saver(var_list)


# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
#                                       sess.graph)
# test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
# tf.global_variables_initializer().run()
    print('hvd rank:', hvd.rank())

    # print('Initialized')


    for epoch in range(500):
        if hvd.rank() == 0:
            train_writer = tf.summary.FileWriter(logdir + '/train/{}'.format(epoch), sess.graph)
        pbar = tg.ProgressBar(n_exp)
        ttl_d_loss = 0
        ttl_g_loss = 0
        for i in range(0, n_exp, batch_size):
            pbar.update(i)
            # _, loss_train = sess.run([train_op, loss_train_sb])
            # ttl_train_loss += loss_train
            # for i in range(3):
                # sess.run(d_clip)
            z_val = np.random.uniform(size=(batch_size, z_dim), low=-1, high=1)
            sess.run(d_loss_op, feed_dict={z_ph:z_val})
            for i in range(3):
                z_val = np.random.uniform(size=(batch_size, z_dim), low=-1, high=1)
                # import pdb; pdb.set_trace()
                sess.run(g_loss_op, feed_dict={z_ph:z_val})
            # sess.run(g_loss_op)
            z_val = np.random.uniform(size=(batch_size, z_dim), low=-1, high=1)
            d_batch_loss = sess.run(d_loss, feed_dict={z_ph:z_val})

            # sess.run(d_clip)
            # g_batch_loss = sess.run(g_loss)
            # g_batch_loss = sess.run(g_loss)
            z_val = np.random.uniform(size=(batch_size, z_dim), low=-1, high=1)
            g_batch_loss = sess.run(g_loss, feed_dict={z_ph:z_val})

            ttl_d_loss += d_batch_loss * batch_size
            ttl_g_loss += g_batch_loss * batch_size

        if hvd.rank() == 0:
            z_val = np.random.uniform(size=(batch_size, z_dim), low=-1, high=1)
            X_g = sess.run(X_sampler_g_sm, feed_dict={z_ph:z_val})
            train_writer.add_summary(X_g)
            train_writer.close()

            save_path = saver.save(sess, "./save/{}/gan_model.ckpt".format(ts))
            print("Model saved in file: %s" % save_path)

        pbar.update(n_exp)
        ttl_d_loss /= n_exp
        ttl_g_loss /= n_exp
        # ttl_train_loss /= n_exp
        # print('epoch {}, train loss {}'.format(epoch, ttl_train_loss))

        print('epoch {}, g loss: {}, d loss: {}'.format(epoch, ttl_g_loss, ttl_d_loss))
        # print(np.mean(loss_train))
        # print(i)
    # import pdb; pdb.set_trace()
    # print(sess.run(names_records['X']))
    coord.request_stop()
    coord.join(threads)

# if __name__ == '__main__':
#     train()
