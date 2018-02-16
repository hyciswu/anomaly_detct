

import tensorflow as tf
from model import classifier, WGAN
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from tensorflow.python.framework import ops
from data import brain
import tensorgraph as tg
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import dicomutils

def plot(arr):
    plt.figure()
    plt.imshow(arr)
    plt.show()

def test_images(shape):
    def slice(folder):
        T2_Ax = ['t2_tse_tra_320_p2', 'Ax T2 FSE', 'Ax T2 FLAIR',
                 'OAx T2 FLAIR', 'AX T2 FRFSE', 't2_tse_tra_P2',
                 'OAx T2 PROPELLER ', 't2_tse_tra_P2_24slice',
                 't2_tse_tra', 'Ax T2 PROPELLER']
        arr = dicomutils.construct3DfromPatient(folder, T2_Ax, resize_shape=tuple(shape), lower=True)
        return arr[:,:,:,np.newaxis]

    bpynz = '/data/tiantan/bpynz/clean/001481800'
    nml = '/data/tiantan/nml/clean/001701437'
    tsjl = '/data/tiantan/tsjl/clean/001550458'
    xxg = '/data/tiantan/svd/clean/001734195'

    imgs = []
    for path in [bpynz, nml, tsjl, xxg]:
        try:
            imgs.append(slice(path))
        except Exception as e:
            print('Error!:', str(e))

    return imgs

# test_images([10,10])


graph = tf.Graph()
with graph.as_default():
# tf.reset_default_graph()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    batch_size = 1
    z_lr = 0.005
    cls_learning_rate = 0.001
    patch_size = [20, 20]
    max_patches = 100
    image_shape = [200, 200]
    z_dim = 100
    # names_records, h, w, c, n_exp = brain(create_tfrecords=False, batch_size=batch_size, shape=[200,200])
    names_records, h, w, c, ttl_exp = brain(create_tfrecords=False, batch_size=batch_size, shape=image_shape)


    image_save_path = './save/image'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    X_ph = tf.placeholder('float32', [None] + image_shape + [1])
    X_gen_patch_ph = tf.placeholder('float32', [None] + patch_size + [1])
    X_patch_ph = tf.placeholder('float32', [None] + patch_size + [1])
    y_patch_ph = tf.placeholder('float32', [None] + patch_size)
    z_ph = tf.placeholder('float32', [None, z_dim])

    with tf.variable_scope('GAN'):
        model = WGAN(h, w, c, z_dim=z_dim, gf_dim=64, df_dim=64)
        g_loss, d_loss, X_gen_sb, img_diff_cost_sb, X_fake_test_sb, y1_fake_test_sb = model.loss(X_real_sb=names_records['X'],
                                                                batch_size=batch_size,
                                                                X_ph = X_ph,
                                                                z_ph = z_ph,
                                                                improved_wgan=True,
                                                                weight_regularize=True)




    h_p, w_p = patch_size
    with tf.variable_scope('Classifier'):
        y_train_sb, y_test_sb = classifier(X_patch_ph, X_gen_patch_ph, h_p, w_p)
        cls_train_cost_sb = tg.cost.entropy(y_train_sb, y_patch_ph)
        cls_valid_cost_sb = tg.cost.entropy(y_test_sb, y_patch_ph)



        # sess.run(init_op)



    optimizer = tf.train.AdamOptimizer(cls_learning_rate)
    update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS, 'Classifier')
    with ops.control_dependencies(update_ops):
        train_op = optimizer.minimize(cls_train_cost_sb)



    # gan_saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     gan_saver.restore(sess, "./save/gan_model.ckpt")
    #     # sess.run(init_op)


    # X_sampler_g_sm = tf.summary.image('X_sampler_g', X_sampler_g, max_outputs=10)

    # Add ops to save and restore only `v2` using the name "v2"

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    cls_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Classifier')
    # cls_saver = tf.train.Saver(cls_var_list)


    # Use the saver object normally after that.
    # with tf.Session() as sess:
    # with tf.Session(graph=graph) as sess:
    sess = tf.Session(graph=graph)
    sess.run(init_op)
        # Initialize v1 since the saver will not.
        # v1.initializer.run()
    report = tf.report_uninitialized_variables()
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'GAN')
    gan_saver = tf.train.Saver(var_list)
    # with tf.Session(graph=graph) as sess:
    gan_saver.restore(sess, "./save/20171202_1226_26537797/gan_model.ckpt")


    sess.run(report)
    print(report)



    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)


    # for epoch in range(100):
    # pbar = tg.ProgressBar(ttl_exp)
    n_exp_ran = 0
    run_train_cost = 0
    run_valid_cost = 0
    for n_iter in range(10000):

        # X_t = generate()
        # print('epoch: {}'.format(epoch))
        ttl_cls_train_cost = 0
        ttl_cls_valid_cost = 0
        best_valid_cost = float('inf')
        # n_exp = 0

        X_val = sess.run(names_records['X'])
        bpynz, nml, tsjl, xxg = test_images(image_shape)
        X_val = bpynz[5:6]
        # X_val = nml
        # plot(dicomutils.seriesFlatten(bpynz[12:13,:,:,0]))

        # X_val[0].shape
        # plot(X_val[0][:,:,0])

        n_exp_ran += len(X_val)

        # for X_val in dataset:
        # z_val = np.random.rand(batchsize, z_dim)
        z_val = np.random.uniform(size=(batch_size, z_dim), low=-1, high=1)
        z_grad_sb = tf.gradients(ys=img_diff_cost_sb, xs=z_ph)
        mom = np.zeros(shape=(batch_size, z_dim))
        print('done')
        epoch = 1
        while True:
            # X_tp1 = X_t + d(mean_cost) / d(z_var)
            #
            # if mse(X_t - X_tp1) < epsilon:
            #     break
            # X_t = X_tp1
            #
            # generate_data()

            # grad, = sess.run(feed_dict={X_ph:X_np})
            # import pdb; pdb.set_trace()
            # print(z_val[0])
            diff_cost = sess.run(img_diff_cost_sb, feed_dict={X_ph:X_val, z_ph:z_val})
            # print(sess.run(y1_fake_test_sb[0][0][0], feed_dict={z_ph:z_val}))
            # print(sess.run(img_diff_cost_sb, feed_dict={z_ph:z_val-0.0001*z_grad}))


            # z_val[0]
            z_grad, = sess.run(z_grad_sb, feed_dict={X_ph:X_val, z_ph:z_val})
            # z_grad[0]

            # z_val[0]
            mom = 0.9 * mom + 0.001 * z_grad
            z_val = z_val - mom

            z_val = (z_val > 1) + (-1 <= z_val) * (z_val <= 1)*z_val + -1 * (z_val < -1)
            z_val = z_val.astype('f4')
            # z_val[0]

            # z_ph
            X_gen_val = sess.run(X_gen_sb, feed_dict={z_ph:z_val})

            epoch += 1
            # print('original')
            # plot(np.hstack([X_val[5,:,:,0], X_gen_val[5,:,:,0], X_val[5,:,:,0]-X_gen_val[5,:,:,0]]))
            if epoch % 10 == 0:
                print('======[{}]======'.format(epoch))
                print('image dif cost:', np.sqrt(np.mean(diff_cost)))
                print('mean grad:', np.sqrt(np.mean(z_grad**2)))
                plot(np.hstack([X_val[0,:,:,0], X_gen_val[0,:,:,0]]))
            # print('generated')
            # plot(X_gen_val[10,:,:,0])
            # plot(X_gen_val[10,:,:,0])
                val = np.mean(z_val ** 2)
                print('mean z val:', np.sqrt(val))

            # if z_grad.length < epsilon:
                # break
        X_gen_val = sess.run(X_gen_sb, feed_dict={z_ph:z_val})
        X_data, X_gen_data, y_data = extract_pos_neg_patches(X_val, X_gen_val,
                                                             patch_size=patch_size,
                                                             max_patches=max_patches)
        X_train, X_valid = tg.utils.split_arr(X_data, train_valid_ratio=[5,1])
        X_gen_train, X_gen_valid = tg.utils.split_arr(X_gen_data, train_valid_ratio=[5,1])
        y_train, y_valid = tg.utils.split_arr(y_data, train_valid_ratio=[5,1])

        train_iter = tg.SequentialIterator(X_train, X_gen_train, y_train, batchsize=32)
        valid_iter = tg.SequentialIterator(X_valid, X_gen_valid, y_valid, batchsize=32)

        print('training classifier')
        n_exp = 0
        for X1_batch, X2_batch, y_batch in train_iter:
            _, cls_train_cost = sess.run([train_op, cls_train_cost_sb], feed_dict={X1_batch, X2_batch, y_batch})
            ttl_cls_train_cost += cls_train_cost * len(X1_batch)
            n_exp += len(X1_batch)
        ttl_cls_train_cost /= float(n_exp)

        run_train_cost = 0.1 * ttl_cls_train_cost + 0.9 * run_train_cost

        print('validating classifier')
        n_exp = 0
        for X1_batch, X2_batch, y_batch in valid_iter:
            cls_train_cost = sess.run([cls_train_cost_sb], feed_dict={X_patch_ph: X1_batch,
                                                                      X_gen_patch_ph: X2_batch,
                                                                      y_patch_ph: y_batch})
            ttl_cls_train_cost += cls_train_cost * len(X1_batch)
            n_exp += len(X1_batch)
        ttl_cls_valid_cost /= float(n_exp)

        run_valid_cost = 0.1 * ttl_cls_valid_cost + 0.9 * run_valid_cost



        # pbar.update(n_exp)


        print('average train cls cost: {}'.format(ttl_cls_train_cost))
        print('average valid cls cost: {}'.format(ttl_cls_valid_cost))

        print('epoch {}. num example {} / {}'.format(n_exp_ran/ttl_exp, n_exp_ran%ttl_exp, ttl_exp))
        if n_iter % 1000 == 0:
            for X_val in test_images():
                z_val = np.random.rand(batchsize, z_dim)
                while True:
                    # X_tp1 = X_t + d(mean_cost) / d(z_var)
                    #
                    # if mse(X_t - X_tp1) < epsilon:
                    #     break
                    # X_t = X_tp1
                    #
                    # generate_data()

                    # grad, = sess.run(feed_dict={X_ph:X_np})
                    z_grad_sb = tf.gradients(img_diff_cost_sb, z_ph)
                    z_grad = sess.run(z_grad_sb, feed_dict={X_ph:X_val, z_ph:z_val})
                    # import pdb; pdb.set_trace()
                    print(np.mean(z_grad))
                    z_val = z_val - z_lr * z_grad

                    # X_gen_val = sess.run(X_gen_sb, feed_dict={z_ph:z_val})

                    # if z_grad.length < epsilon:
                    #     break
                X_gen_val = sess.run(X_gen_sb, feed_dict={z_ph:z_val})

                test_patches = extract_patches_2d(X_gen_val, patch_size=patch_size)

                y_valid = sess.run(y_test_sb, feed_dict={X_patch_ph:test_patches})
                confidence_mask = reconstruct_from_patches_2d(y_valid, image_size=image_shape)

                cv2.imwrite(confidence_mask, image_save_path+'/mask_{}.png'.format(n_iter))
                cv2.imwrite(X_val, image_save_path+'/orig_{}.png'.format(n_iter))
                cv.imwrite(X_val - X_gen_val, image_save_path+'/diff_{}.png'.format(n_iter))

            if run_valid_cost < best_valid_cost:
                best_valid_cost = run_valid_cost
                cls_saver.save("./save/cls_model.ckpt")






# def extract_patch(X, patch_size):
#     import cv2
#     from sklearn.feature_extraction.image import extract_patches_2d
#     from sklearn.feature_extraction.image import reconstruct_from_patches_2d
#     import matplotlib.pyplot as plt
#     X1 = cv2.imread('./images/img1.png', 0)
#     X1_h = X1[200:1200, 200:1200]
#     X1_h.shape
#     X_patch = extract_patches_2d(X1_h, patch_size=(100,100))
#     X = reconstruct_from_patches_2d(X_patch, image_size=(1000,1000))

#
# def reconstruct_patch(patch, image):
#     plt.figure()
#     plt.imshow(X1_h)
#     plt.show()
#
#     plt.figure()
#     plt.imshow(X)
#     plt.show()


def extract_pos_neg_patches(X, X_gen, patch_size=(20,20), max_patches=100):
    def batch_make_patches(X, patch_size, max_patches):
        X_r = np.rollaxis(X, 0, 3)
        X_patch = extract_patches_2d(X_r, patch_size=patch_size,
                                     max_patches=max_patches,
                                     random_state=1012)

        X_back = np.rollaxis(X_patch, 3, 1)
        # X_back.shape
        n,b,h,w = X_back.shape
        X_back = X_back.reshape(n*b, h, w)
        return X_back



    # X1 = cv2.imread('./images/img1.png', 0)
    # X1_h = X1[200:1200, 200:1200]
    # X1_h.shape
    #
    # X2 = cv2.imread('./images/img2.png', 0)
    # X2_h = X2[200:1200, 200:1200]
    # X = np.stack([X1_h, X2_h])

    X_patch = batch_make_patches(X, patch_size=patch_size, max_patches=max_patches)
    X_gen_patch = batch_make_patches(X_gen, patch_size=patch_size, max_patches=max_patches)


    pos = np.ones(len(X_back))
    neg = np.zeros(len(X_back))
    idx = np.arange(len(X_back))
    np.random.shuffle(idx)
    same = [X_patch, X_gen_patch]
    diff = [X_patch, X_gen_patch[ridx]]

    X = np.concatenate([X_patch, X_patch])
    X_gen = np.concatenate([X_gen_patch, X_gen_patch[rdix]])
    y = np.concatenate([pos, neg])

    ridx = np.arange(len(X))
    np.random.shuffle(ridx)
    X = X[ridx]
    X_gen = X_gen[ridx]
    y = y[ridx]
    return X, X_gen, y

#
# def rolling_window(arr, window):
#     """Very basic multi dimensional rolling window. window should be the shape of
#     of the desired subarrays. Window is either a scalar or a tuple of same size
#     as `arr.shape`.
#     """
#     shape = np.array(arr.shape*2)
#     strides = np.array(arr.strides*2)
#     window = np.asarray(window)
#     shape[arr.ndim:] = window # new dimensions size
#     shape[:arr.ndim] -= window - 1
#     if np.any(shape < 1):
#         raise ValueError('window size is too large')
#     return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
#
#
# import matplotlib.pyplot as plt
# im = plt.imread('/Users/hycis/data/img.png')
# im.strides
# h, w = im[:,:,0].shape
# im.shape
# im2 = im[200:1400,150:1350,0]
# im2.shape
# im2.strides
# import numpy as np
# arrs = np.split(im2, 10)
# arrs[0].shape
# im3 = im2.reshape(100, 1200*1200/100)
#
# im2
# out = rolling_window(im2, [300,300])
# out.shape
#
# import cv2
# im4 = cv2.resize(im2, (200,200))
#
# from sklearn.feature_extraction.image import extract_patches_2d
#
# im.shape
# patches = extract_patches_2d(im, patch_size=(20,20), max_patches=100, random_state=1012)
# patches.shape
# patches[0].shape
#
# patches = extract_patches_2d(im4, patch_size=(20,20), max_patches=100, random_state=1012)
# plt.figure()
# plt.imshow(patches[3])
# plt.show()
#
# patches = extract_patches_2d(im4, patch_size=(20,20), max_patches=100, random_state=1012)
# plt.figure()
# plt.imshow(patches[3])
# plt.show()
#
#
#
#
# plt.figure()
#         # X_np = sess.run(feed_dict={z_ph: z_new})
#
#
#     # print("v1 : %s" % v1.eval())
#     # print("v2 : %s" % v2.eval())
