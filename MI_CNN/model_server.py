# coding=utf-8
"""
识别模型server,包括填空题批改/改分改错识别/数字识别
"""
import numpy as np
import cv2
import math
from scipy import ndimage
import tensorflow as tf
from signal_preprocessing import average_fft_of_singal_trail
feature_d = 26
cata_count = 3

class gap_recognition_server:
    def __init__(self, saved_model):
        self._model_init(saved_model)

    def _model_init(self, saved_model):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        def cnn_layer(x, filter_shape, max_pool=True):
            W = weight_variable(filter_shape)
            b = bias_variable([filter_shape[3]])
            h = tf.nn.relu(conv2d(x, W) + b)
            if max_pool:
                h = max_pool_2x2(h)
            return h, W
        with tf.name_scope('MI-3'):
            with tf.name_scope('placeholder'):
                self.x = tf.placeholder(tf.float32, [None, 2, feature_d, 1])
            with tf.name_scope('CONV-1'):
                # 1.CNN layer
                cnn_1, w1 = cnn_layer(self.x, [1, 4, 1, 32], max_pool=False)
            with tf.name_scope('CONV-2'):
                # 2.CNN layer
                cnn_2, w2 = cnn_layer(cnn_1, [1, 4, 32, 16], max_pool=False)
            with tf.name_scope('FC_1'):
                # 3.FC_1
                d = 256
                W_fc1 = weight_variable([2 * (feature_d) * 16, d])
                b_fc1 = bias_variable([d])
                h_pool5_flat = tf.reshape(cnn_2, [-1, 2 * (feature_d) * 16])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
                keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                # h_fc1_drop = h_fc1
            with tf.name_scope('FC_2'):
                # 4.FC_2
                W_fc2 = weight_variable([d, cata_count])
                b_fc2 = bias_variable([cata_count])
                self.y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
                self.prediction = tf.argmax(self.y, 1)

            # all_vars = tf.all_variables()
            # gap_vars = [k for k in all_vars if k.name.startswith('gap')]
            saver = tf.train.Saver()
            self.sess = tf.Session()
            saver.restore(self.sess, saved_model)

        return

    def predict(self, data_frame):
        """
        :param data_frame: 400个两通道序列
        :return:
        """
        data_f = average_fft_of_singal_trail(data_frame)
        return self.prediction.eval(feed_dict={self.x: data_f.reshape(-1, 2, feature_d, 1)}, session=self.sess)[0]


# load model
gap_model = gap_recognition_server('saved_model/gap/saved_model-9000')

