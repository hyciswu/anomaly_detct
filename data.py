

import tensorgraph as tg
import dicomutils
import os
import numpy as np
import tensorflow as tf


def cifar10(create_tfrecords=True, batch_size=32):
    tfrecords = tg.utils.MakeTFRecords()
    tfpath = './cifar10.tfrecords'
    if not os.path.exists(tfpath):
        X_train, y_train, X_test, y_test = tg.dataset.Cifar10()
        # import pdb; pdb.set_trace()
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X_train, 'y':y_train}, save_path=tfpath)
    n_exp = sum(1 for _ in tf.python_io.tf_record_iterator(tfpath))
    names_records = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath],
                                              data_shapes={'X':[32,32,3], 'y':[10]},
                                              batch_size=batch_size)
    return dict(names_records), 32, 32, 3, n_exp


def mnist(create_tfrecords=True, batch_size=32):
    tfrecords = tg.utils.MakeTFRecords()
    tfpath = './mnist.tfrecords'
    if not os.path.exists(tfpath):
        X_train, y_train, X_test, y_test = tg.dataset.Mnist()
        tg.preprocess.random_shift(X_train)
        import pdb; pdb.set_trace()
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X_train, 'y':y_train}, save_path=tfpath)
    n_exp = sum(1 for _ in tf.python_io.tf_record_iterator(tfpath))
    names_records = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath],
                                              data_shapes={'X':[28,28,1], 'y':[10]},
                                              batch_size=batch_size)
    return dict(names_records), 28, 28, 1, n_exp


def brain(create_tfrecords=True, batch_size=32, shape=[100,100]):
    h, w = shape
    tfrecords = tg.utils.MakeTFRecords()
    tfpath = './brain_{}x{}.tfrecords'.format(h,w)
    if not os.path.exists(tfpath):

        high_topdir = '/data/tiantan/normal/clean/'


        # import dicomutils
        dirs = dicomutils.getPatientDirectories(high_topdir)

        high_topdir2 = '/data/tiantan/normal_brain/'
        dirs2 = dicomutils.getPatientDirectories(high_topdir2)
        dirs = dirs + dirs2
        print('total ({}) dirs'.format(len(dirs)))

        T2_Ax = ['t2_tse_tra_320_p2', 'Ax T2 FSE', 'Ax T2 FLAIR',
                 'OAx T2 FLAIR', 'AX T2 FRFSE', 't2_tse_tra_P2',
                 'OAx T2 PROPELLER ', 't2_tse_tra_P2_24slice',
                 't2_tse_tra', 'Ax T2 PROPELLER']
        arrs = []
        for folder in dirs:
            try:
                arr = dicomutils.construct3DfromPatient(folder, T2_Ax, resize_shape=(h,w), lower=True)
                assert len(arr) > 20
                arr = arr[5:15]
                arrs.append(arr)
            except Exception as e:
                print('Error!:', str(e))
        # import numpy as np
        arrs = np.concatenate(arrs)

        miu = arrs.mean(axis = (1,2))
        std = arrs.std(axis = (1,2))
        arrs_norm = (arrs - miu[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]

        # arrs = arrs / arrs.max()
        # flat = dicomutils.seriesFlatten(arrs_norm)

        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.imshow(flat)
        # plt.show()


        # import tensorgraph as tg
        # arrs = np.expand_dims(arrs, -1)
        #
        # r_arrs = tg.dataset.preprocess.random_rotation(arrs, rg=30, row_axis=1,
        #                                        col_axis=2, channel_axis=0,
        #                                        fill_mode='nearest', cval=0.)
        #
        # plt.figure()
        # plt.imshow(dicomutils.seriesFlatten(arrs[:10]))
        # plt.show()
        # plt.figure()
        # plt.imshow(dicomutils.seriesFlatten(r_arrs[:49]))
        # plt.show()


        # z_arrs = tg.preprocess.random_zoom(arrs, zoom_range=[0.5, 0.5], row_axis=1,
        #                                    col_axis=2, channel_axis=0,
        #                                    fill_mode='nearest', cval=0.)
        #
        # # import pdb; pdb.set_trace()
        # arrs = np.concatenate([arrs, z_arrs, r_arrs])


        print('======', arrs.shape)
        tfrecords.make_tfrecords_from_arrs(data_records={'X':arrs}, save_path=tfpath)
    n_exp = sum(1 for _ in tf.python_io.tf_record_iterator(tfpath))
    names_records = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath],
                                              data_shapes={'X':[h,w,1]},
                                              batch_size=batch_size)

    return dict(names_records), h, w, 1, n_exp





# def generate_pos_neg(X_val, X_gen_val):
#
#
#
#
#
#
# X1_data, X2_data, y_data = generate_pos_neg(X_val, X_gen_val)
