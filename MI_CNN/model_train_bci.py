# coding=utf-8
# ==================================================================
# to train and export model for recognition
#
# input:
#       data: process images with image_preprocessing function
#               and stack all the returns into a matrix of N*784(N for quantity of image)
#       label: digital number
# output:
#        saved model
# ==================================================================
"""
训练自己的脑电数据,两类或者三类
"""
# import modules
import tensorflow as tf
import os
from data_prepare import *

# hypeparamaters
# ============================================================================================
tf.flags.DEFINE_integer("reg", 0.01, "l2 regulization streght")
tf.flags.DEFINE_integer("batch_size", 128, "training batch size")
tf.flags.DEFINE_integer("lr", 1e-1, "learning rate")
tf.flags.DEFINE_integer("iter", 1000000, "training iterations")
tf.flags.DEFINE_integer("save_step", 100000, "save the model every save_step")
tf.flags.DEFINE_integer("decade", 10000, "learning rate decade")
tf.flags.DEFINE_string("restore_dir", None, "restore directory")

# prepare data
train_labels = np.load('data/npy/train/train_labels.npy')
train_data_ = np.load('data/npy/train/train_data.npy')
train_data = data_class(train_data_, train_labels)
test_data_ = np.load('data/npy/test/test_data.npy')
test_labels = np.load('data/npy/test/test_labels.npy')
test_data = data_class(test_data_, test_labels)
print 'train and test data:'
print 'train data:'
print train_data_.shape
print 'test data:'
print test_data_.shape
# paras
input_shape = [40, 20, 2]
cata_count = 2
filter_num = [32, 32]
cnn_filter_shape = [[5, 10], [5, 10]]
d = 1024    # FC W.SIZE
save_path = os.path.abspath(os.path.join(os.path.curdir, "saved_model/"))
log_dir = os.path.abspath(os.path.join(os.path.curdir, "log/"))
# clear log data
file_list = os.listdir('log/')
if len(file_list) > 0:
    for item in file_list:
        os.remove('log/'+item)
FLAGS = tf.flags.FLAGS

# self define methods
# ============================================================================================
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')


def cnn_layer(x, filter_shape, max_pool=True):
    W = weight_variable(filter_shape)
    b = bias_variable([filter_shape[3]])
    h = tf.nn.relu(conv2d(x, W) + b)
    if max_pool:
        h = max_pool_1x2(h)
    return h, W


# Create the model
# ============================================================================================

# placehodlers
# ============================================================================================
# input
with tf.Graph().as_default():
    with tf.name_scope('MI-bci'):
        with tf.name_scope('placeholder'):
            x = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]])
            # output
            y_ = tf.placeholder(tf.float32, [None, cata_count])
        with tf.name_scope('CONV-1'):
            # 1.CNN layer
            cnn_1, w1 = cnn_layer(x, [cnn_filter_shape[0][0], cnn_filter_shape[0][1], 2, filter_num[0]])
        with tf.name_scope('CONV-2'):
            # 2.CNN layer
            cnn_2, w2 = cnn_layer(cnn_1, [cnn_filter_shape[1][0], cnn_filter_shape[1][1], filter_num[0], filter_num[1]])
        with tf.name_scope('FC_1'):
            # 3.FC_1
            W_fc1 = weight_variable([input_shape[0]*input_shape[1]*filter_num[1]/16, d])
            b_fc1 = bias_variable([d])
            h_pool5_flat = tf.reshape(cnn_2, [-1, input_shape[0]*input_shape[1]*filter_num[1]/16])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            # h_fc1_drop = h_fc1
        with tf.name_scope('FC_2'):
            # 4.FC_2
            W_fc2 = weight_variable([d, cata_count])
            b_fc2 = bias_variable([cata_count])
            y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # Define loss and optimizer
        with tf.name_scope('loss'):
            re_loss = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
            # prevent NaN when training
            cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))) + FLAGS.reg * re_loss

        train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # summary
        with tf.name_scope('summaries'):
            loss_summary = tf.summary.scalar('cross_entropy', cross_entropy)
            acc_summary = tf.summary.scalar('accuracy', accuracy)
            test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])

        # saver
        saver = tf.train.Saver(tf.all_variables())
        # summary
        summary_writer = tf.summary.FileWriter(log_dir)

        # import data
        test_batch = test_data.next_batch(150)
        f_dict_for_test = {x: test_batch[0], y_: test_batch[1], keep_prob: 1}
        # train and validation
        # ============================================================================================
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            print "initializing variables\n"
            summary_writer.add_graph(sess.graph)
            if FLAGS.restore_dir:
                saver.restore(sess, FLAGS.restore_dir)
                pre_step = int(FLAGS.restore_dir.split('-')[1])
            else:
                pre_step = 0
                sess.run(init)
            # training
            # ============================================================================================
            print 'start training...\n'
            for step in range(FLAGS.iter):
                batch = train_data.next_batch(FLAGS.batch_size)
                feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
                if step % 200 == 0:
                    train_accuracy = accuracy.eval(feed_dict=feed_dict)
                    print("step %d, training accuracy %g" % (step+pre_step, train_accuracy))
                    test_accuracy = accuracy.eval(feed_dict=f_dict_for_test)
                    test_s = test_acc_summary.eval(feed_dict=f_dict_for_test)
                    summary_writer.add_summary(test_s, step)
                    print("step %d, test accuracy %g" % (step+pre_step, test_accuracy))
                if step != 0 | step % FLAGS.save_step == 0:
                    path = saver.save(sess, save_path, global_step=step + pre_step)
                    print 'save file to %s' % path
                if step != 0 | step % FLAGS.decade == 0:
                    FLAGS.lr *= 0.1
                summaries, _, loss = sess.run([train_summary_op, train_step, cross_entropy], feed_dict)
                summary_writer.add_summary(summaries, step+pre_step)

                # tr, loss = sess.run([train_step, cross_entropy], feed_dict)
                print 'step-%s:%s' % ((step+pre_step), loss)


            # testing
            # ============================================================================================
            # print 'final result...\n'
            # print("test accuracy %g" % accuracy.eval(feed_dict=f_dict))



