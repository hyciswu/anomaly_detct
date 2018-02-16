
import tensorgraph as tg
from tensorgraph.utils import same_nd
import numpy as np
import tensorflow as tf
from tensorgraph.layers import Conv2D, RELU, MaxPooling, LRN, Tanh, Dropout, \
                               Softmax, Flatten, Linear, TFBatchNormalization, AvgPooling, \
                               Lambda, Reshape, Conv2D_Transpose, LeakyRELU, Sigmoid, ReduceMax, \
                               BatchNormalization, Concat
from tensorgraph.utils import same, valid
import tensorflow.contrib as tc


def allcnn(nclass, h, w, c):
    with tf.name_scope('Cifar10AllCNN'):
        seq = tg.Sequential()
        seq.add(Conv2D(input_channels=c, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b1'))
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))

        seq.add(Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(2, 2), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b3'))
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))

        seq.add(Conv2D(input_channels=96, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b5'))
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(2, 2), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b7'))
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(1, 1), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=192, num_filters=nclass, kernel_size=(1, 1), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b9'))
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))

        seq.add(AvgPooling(poolsize=(h, w), stride=(1,1), padding='VALID'))
        seq.add(Flatten())
        seq.add(Softmax())
    return seq


def classifier(X_ph, X_gen_ph, h, w):
    with tf.variable_scope('Classifier'):
        X_sn = tg.StartNode(input_vars=[X_ph])
        X_gen_sn = tg.StartNode(input_vars=[X_gen_ph])
        h1, w1 = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        h2, w2 = same(in_height=h1, in_width=w1, stride=(2,2), kernel_size=(2,2))
        h3, w3 = same(in_height=h2, in_width=w2, stride=(1,1), kernel_size=(3,3))
        h4, w4 = same(in_height=h3, in_width=w3, stride=(2,2), kernel_size=(2,2))

        print('---', h, w)
        X_hn = tg.HiddenNode(prev=[X_sn],
                             layers=[Conv2D(input_channels=1, num_filters=32, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
                                     BatchNormalization(input_shape=[h1,w1,32]),
                                     RELU(),
                                     MaxPooling(poolsize=(2, 2), stride=(2,2), padding='SAME'),
                                     LRN(),
                                     Conv2D(input_channels=32, num_filters=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
                                     BatchNormalization(input_shape=[h3,w3,64]),
                                     RELU(),
                                     MaxPooling(poolsize=(2, 2), stride=(2,2), padding='SAME'),
                                     Flatten(),
                                     ])

        X_gen_hn = tg.HiddenNode(prev=[X_gen_sn],
                                 layers=[Conv2D(input_channels=1, num_filters=32, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
                                         BatchNormalization(input_shape=[h1,w1,32]),
                                         RELU(),
                                         MaxPooling(poolsize=(2, 2), stride=(2,2), padding='SAME'),
                                         LRN(),
                                         Conv2D(input_channels=32, num_filters=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
                                         BatchNormalization(input_shape=[h3,w3,64]),
                                         RELU(),
                                         MaxPooling(poolsize=(2, 2), stride=(2,2), padding='SAME'),
                                         Flatten(),
                                         ])

        print('===', h4*w4*64*2)

        merge_hn = tg.HiddenNode(prev=[X_hn, X_gen_hn], input_merge_mode=Concat(),
                                 layers=[Linear(h4*w4*64*2, 100),
                                         RELU(),
                                         BatchNormalization(input_shape=[100]),
                                         Linear(100, 1),
                                         Sigmoid()])


        en = tg.EndNode(prev=[merge_hn])

        graph = tg.Graph(start=[X_sn, X_gen_sn], end=[en])
        y_train, = graph.train_fprop()
        y_test, = graph.test_fprop()
    return y_train, y_test


class WGAN(object):

    def __init__(self, h, w, c, z_dim=100, gf_dim=64, df_dim=64):


        self.z_dim = z_dim


        out_shape2 = same_nd([h, w], kernel_size=(5,5), stride=(2,2))
        out_shape4 = same_nd(out_shape2, kernel_size=(5,5), stride=(2,2))
        out_shape8 = same_nd(out_shape4, kernel_size=(5,5), stride=(2,2))
        out_shape16 = same_nd(out_shape8, kernel_size=(5,5), stride=(2,2))
        h16, w16 = out_shape16


        with tf.variable_scope('Generator'):
            self.g_layers = [Linear(z_dim, 8*gf_dim*h16*w16),
                           Reshape([-1, h16, w16, 8*gf_dim]),
                           # TFBatchNormalization(name='gbn1'),
                           BatchNormalization(input_shape=[h16, w16, 8*gf_dim]),
                           RELU(),
                           Conv2D_Transpose(input_channels=8*gf_dim, num_filters=4*gf_dim,
                                            output_shape=out_shape8, kernel_size=(5,5), stride=(2,2),
                                            padding='SAME'),
                           # TFBatchNormalization(name='gbn2'),
                           BatchNormalization(input_shape=out_shape8+[4*gf_dim]),
                           RELU(),
                           Conv2D_Transpose(input_channels=4*gf_dim, num_filters=2*gf_dim,
                                            output_shape=out_shape4, kernel_size=(5,5), stride=(2,2),
                                            padding='SAME'),
                           # TFBatchNormalization(name='gbn3'),
                           BatchNormalization(input_shape=out_shape4+[2*gf_dim]),
                           RELU(),
                           Conv2D_Transpose(input_channels=2*gf_dim, num_filters=gf_dim,
                                            output_shape=out_shape2, kernel_size=(5,5), stride=(2,2),
                                            padding='SAME'),
                           # TFBatchNormalization(name='gbn4'),
                           BatchNormalization(input_shape=out_shape2+[gf_dim]),
                           RELU(),
                           Conv2D_Transpose(input_channels=gf_dim, num_filters=c,
                                            output_shape=(h, w), kernel_size=(5,5), stride=(2,2),
                                            padding='SAME'),
                           # Sigmoid()
                           ]


        out_shape2 = same_nd([h, w], kernel_size=(5,5), stride=(2,2))
        out_shape4 = same_nd(out_shape2, kernel_size=(5,5), stride=(2,2))
        out_shape8 = same_nd(out_shape4, kernel_size=(5,5), stride=(2,2))
        out_shape16 = same_nd(out_shape8, kernel_size=(5,5), stride=(2,2))
        h16, w16 = out_shape16

        with tf.variable_scope('Discriminator'):
            self.d1_layers = [Conv2D(input_channels=c, num_filters=df_dim,
                                  kernel_size=(5,5), stride=(2,2), padding='SAME'),
                           LeakyRELU(),
                           Conv2D(input_channels=df_dim, num_filters=2*df_dim,
                                  kernel_size=(5,5), stride=(2,2), padding='SAME'),
                           ]
                           # TFBatchNormalization(name='dbn1'),
            self.d2_layers = [
                           BatchNormalization(input_shape=out_shape4+[2*df_dim]),
                           LeakyRELU(),
                           Conv2D(input_channels=2*df_dim, num_filters=4*df_dim,
                                  kernel_size=(5,5), stride=(2,2), padding='SAME'),
                            ]

            self.d3_layers = [

                           # TFBatchNormalization(name='dbn2'),
                           BatchNormalization(input_shape=out_shape8+[4*df_dim]),
                           LeakyRELU(),
                           Conv2D(input_channels=4*df_dim, num_filters=8*df_dim,
                                  kernel_size=(5,5), stride=(2,2), padding='SAME'),
                            ]
            self.d4_layers = [
                           # TFBatchNormalization(name='dbn3'),
                           BatchNormalization(input_shape=out_shape16+[8*df_dim]),
                           LeakyRELU(),
                           ReduceMax(reduction_indices=[1,2]),
                           ]
            self.d5_layers = [Flatten(),
                           Linear(8*df_dim, 1),
                           # LeakyRELU(),
                           # Linear(1000, 1)
                        #    Sigmoid()
                           ]
            print('====:', 8*df_dim)


    def generator(self, z_fake):
        z_fake_sn = tg.StartNode(input_vars=[z_fake])
        hn = tg.HiddenNode(prev=[z_fake_sn], layers=self.g_layers)
        en = tg.EndNode(prev=[hn])
        graph = tg.Graph(start=[z_fake_sn], end=[en])
        X_fake_train_sb, = graph.train_fprop()
        X_fake_test_sb, = graph.test_fprop()
        return X_fake_train_sb, X_fake_test_sb


    def discriminator(self, X):
        X_sn = tg.StartNode(input_vars=[X])
        hn1 = tg.HiddenNode(prev=[X_sn], layers=self.d1_layers)
        hn2 = tg.HiddenNode(prev=[hn1], layers=self.d2_layers)
        hn3 = tg.HiddenNode(prev=[hn2], layers=self.d3_layers)
        hn4 = tg.HiddenNode(prev=[hn3], layers=self.d4_layers)
        hn5 = tg.HiddenNode(prev=[hn4], layers=self.d5_layers)
        en1 = tg.EndNode(prev=[hn1])
        en2 = tg.EndNode(prev=[hn2])
        en3 = tg.EndNode(prev=[hn3])
        en4 = tg.EndNode(prev=[hn4])
        en5 = tg.EndNode(prev=[hn5])
        graph = tg.Graph(start=[X_sn], end=[en1,en2,en3,en4,en5])
        y1, y2, y3, y4, y5 = graph.train_fprop()
        y1_test, y2_test, y3_test, y4_test, y5_test = graph.test_fprop()
        return y1, y2, y3, y4, y5, y1_test, y2_test, y3_test, y4_test, y5_test


    def classfication(self, X):
        X_sn = tg.StartNode(input_vars=[X])
        hn = tg.HiddenNode(prev=[X_sn],
                           layers=[])
        en = tg.EndNode(prev=[hn])
        graph = tg.Graph(start=[X_sn], end=[en])
        y_train, = graph.train_fprop()
        y_test, = graph.test_fprop()
        return y_train, y_test


    def loss(self, X_real_sb, batch_size, X_ph, z_ph, scale=1, improved_wgan=True, weight_regularize=True):

        # z_fake = tf.random_uniform([batch_size, self.z_dim], minval=-1, maxval=1)

        X_fake_sb, X_fake_test_sb = self.generator(z_ph)
        y1_fake_sb, y2_fake_sb,_,_, y_fake_sb, \
        y1_fake_test_sb, y2_fake_test_sb, y3_fake_test_sb, y4_fake_test_sb, \
        y_fake_test_sb = self.discriminator(X_fake_sb)
        y1_real_sb, y2_real_sb, y3_real_sb, y4_real_sb, y_real_sb, _, _, _, _, _ = self.discriminator(X_real_sb)
        _, _, _, _, _, y1_real_test_sb, y2_real_test_sb, y3_real_test_sb, y4_real_test_sb, y_real_test_sb = self.discriminator(X_ph)

        # z_var = tf.Variable(np.zeros(batch_size, self.z_dim))
        # X_gen_sb = self.generator(z_var)

        with tf.name_scope('Loss'):
            g_loss = tf.reduce_mean(y_fake_sb)
            d_loss = tf.reduce_mean(y_real_sb - y_fake_sb)

            if improved_wgan:
                epsilon = tf.random_uniform([], 0.0, 1.0)
                X_hat = epsilon * X_real_sb + (1 - epsilon) * X_fake_sb
                # X_hat_sn = tg.StartNode(input_vars=[x_hat])
                d1_hat, d2_hat, _,_,d_hat, _,_,_,_,_ = self.discriminator(X_hat)

                # graph = tg.Graph(start=[X_hat_sn], end=[y_en])
                # d_hat, = graph.train_fprop()
                #
                ddx = tf.gradients(d_hat, X_hat)[0]
                ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
                ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
                d_loss = d_loss + ddx

            if weight_regularize:
                d_reg = tc.layers.apply_regularization(
                            tc.layers.l1_regularizer(2.5e-5),
                            weights_list=[var for var in tf.global_variables() if 'Discriminator' in var.name]
                            )
                d_loss = d_loss + d_reg

                g_reg = tc.layers.apply_regularization(
                            tc.layers.l1_regularizer(2.5e-5),
                            weights_list=[var for var in tf.global_variables() if 'Generator' in var.name]
                            )
                g_loss = g_loss + g_reg

            # self.g_loss_reg = self.g_loss + self.reg
            # self.d_loss_reg = self.d_loss + self.reg


        orig_diff = tg.cost.mse(X_fake_test_sb, X_ph)
        orig_diff = tf.sqrt(orig_diff)
        # orig_diff = tf.sqrt(orig_diff) + tf.exp(tf.clip_by_value(orig_diff,0,10)/3) - 1

        y1_cost = tg.cost.mse(y1_fake_test_sb, y1_real_test_sb)
        y2_cost = tg.cost.mse(y2_fake_test_sb, y2_real_test_sb)
        y3_cost = tg.cost.mse(y3_fake_test_sb, y3_real_test_sb)
        y4_cost = tg.cost.mse(y4_fake_test_sb, y4_real_test_sb)


        # img_diff_cost = img_diff_cost + img_d_cost
        # img_diff_cost = y1_cost * 0.1 + y2_cost * 0.2 + y3_cost * 0.3 + y4_cost

        # y1_cost = tf.sqrt(y1_cost)
        # y1_cost = tf.sqrt(y1_cost) + tf.exp(tf.clip_by_value(y1_cost,0,10)/3) - 1
        # y2_cost = tf.sqrt(y2_cost)
        # y2_cost = tf.sqrt(y2_cost) + tf.exp(tf.clip_by_value(y2_cost,0,10)/3) - 1
        # img_diff_cost = orig_diff * 0.2 + y1_cost * 0.2 + y2_cost

        # img_diff_cost = orig_diff


        # img_diff_cost = y1_cost + 0.05 * y2_cost - y_real_test_sb
        img_diff_cost = y1_cost + 0.05 * y2_cost
        # img_diff_cost = -y_real_test_sb

        # img_diff_cost = y1_cost * 0.3 + y2_cost
        # img_diff_cost = y1_cost * 0.05 + y2_cost + y3_cost + 0.1 * y4_cost
        # + y3_cost * 0.1
        # y1_cost 940
        # y2_cost 24
        # y3_cost 7
        # y4_cost 200
        # img_diff_cost = tg.cost.mse(y2_fake_test_sb, y2_real_test_sb)
        return g_loss, d_loss, X_fake_sb, img_diff_cost, X_fake_test_sb, y1_fake_test_sb



# import tensorflow as tf
# x = tf.placeholder('float32', [10, 10])
# z = tf.ones((10,10)) + x
# tf.gradients(z, x)
