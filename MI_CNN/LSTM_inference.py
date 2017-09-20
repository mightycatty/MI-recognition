"""
LSTM FOR BCI DATASET IV 2B
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import os
from signal_preprocessing import feature_extract



class lstm_server():
    def __init__(self, saved_model):
        self._model_init(saved_model)

    def _model_init(self, model_dir):
        # Network Parameters
        n_input = 80  # MNIST data input (img shape: 28*28)
        n_steps = 20  # timesteps
        n_hidden = 16  # hidden layer num of features
        n_classes = 2  # MNIST total classes (0-9 digits)

        def RNN(x, weights, biases):
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

            # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.unstack(x, n_steps, 1)

            # Define a lstm cell with tensorflow
            lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

            # Get lstm cell output
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

            # Linear activation, using rnn inner loop last output
            return tf.matmul(outputs[-1], weights['out']) + biases['out']

        with tf.name_scope('LSTM'):
            # tf Graph input
            self.x = tf.placeholder("float", [None, n_steps, n_input])
            # Define weights
            weights = {
                'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
            }
            biases = {
                'out': tf.Variable(tf.random_normal([n_classes]))
            }
            pred = RNN(self.x, weights, biases)
            # Define loss and optimizer
            # Evaluate model
            self.correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # saver
        saver = tf.train.Saver(tf.all_variables())
        self.sess = tf.Session()
        saver.restore(self.sess, model_dir)
        return

    def predict(self, data):
        data = feature_extract(data)
        return self.correct_pred.eval(feed_dict={self.x: data}, session=self.sess)[0]


model_dir = ''
lstm_server_ob = lstm_server(model_dir)
