import time
import shutil
import os
import tensorflow as tf
import scipy.io as sc
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from scipy.signal import butter, lfilter
import xgboost as xgb
from data_getter import get_next_train_batch, get_next_test_batch, get_feature_number, get_batch_size, get_n_step


# tensorboard things
logfile = "log"
if os.path.exists(logfile):
    shutil.rmtree(logfile)


def i_want_to_see(title='title', content=[]):
    '''
    print what you want to see in a cool way
    @param name:
    @param content:
    '''

    print('------------------------------ [ ',
          title, ' ] ------------------------------')
    for c in content:
        print(c)


def weight_variable(shape):
    '''
    @param
    @return
    '''
    initial = tf.random_normal(shape)
    return tf.Variable(initial, trainable=True)


def bias_variable(shape):
    '''
    @param
    @return
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=True)


def apply_fully_connect(x, x_size, fc_size):
    fc_weight = weight_variable([x_size, fc_size])
    fc_bias = bias_variable([fc_size])
    return tf.add(tf.matmul(x, fc_weight), fc_bias)


def my_model(X):
    '''
    model structure
    @return predict result
    '''

    # neurons in hidden layer
    input_layer_units = 64
    # fc1_layer_nuits = 128
    # fc2_layer_nuits = 128
    # fc3_layer_nuits = 256
    # fc4_layer_nuits = 256
    # fc5_layer_nuits = 256

    X = tf.reshape(X, [-1, n_inputs])
    input_layer = tf.nn.relu(apply_fully_connect(
        x=X, x_size=n_inputs, fc_size=input_layer_units))

    # -------------------------------- lstm --------------------------------
    X_in = tf.reshape(input_layer, [-1, n_step, input_layer_units])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        input_layer_units, forget_bias=1, state_is_tuple=True)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(
        lstm_cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    lstm_output = outputs[-1]
    # -------------------------------- lstm --------------------------------

    output_layer = tf.nn.softmax(apply_fully_connect(
        x=lstm_output, x_size=input_layer_units, fc_size=n_classes))

    return output_layer

######################################################################################################################################################


# about data
n_step = get_n_step()
n_inputs = feature_number = get_feature_number()
batch_size = get_batch_size()
n_classes = 3  # there are 3 different kind of classes

# about model
lameda = 0.001
train_times = 200000
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate = tf.train.exponential_decay(
    learning_rate=0.01,
    global_step=global_step,
    decay_steps=10,
    decay_rate=0.999,
    staircase=False,
    name=None
)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_step, feature_number], name="features")
y = tf.placeholder(tf.float32, [None, n_classes], name="labels")
# keep_prob = tf.placeholder(tf.float32, name='keep_prob')

pred = my_model(x)

# L2 loss prevents this overkill neural network to overfit the data
l2 = lameda * sum(tf.nn.l2_loss(tf_var)
                  for tf_var in tf.trainable_variables())

with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=pred, labels=y))+l2  # Softmax loss
    summary_loss = tf.summary.scalar("loss", cost)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(
        learning_rate).minimize(cost, global_step=global_step)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    summary_accuracy = tf.summary.scalar("accuracy", accuracy)

merge = tf.summary.merge_all()


init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logfile+"/train", sess.graph)
    test_writer = tf.summary.FileWriter(logfile+"/test", sess.graph)

    test_x, test_y = get_next_test_batch()

    for step in range(train_times):

        sess.run(tf.local_variables_initializer())
        ################ train #################
        batch_x, batch_y = get_next_train_batch()
        sess.run([train_op], feed_dict={
            x: batch_x,
            y: batch_y,
            # keep_prob: 0.5,
        })

        ############### print something ##################
        if step % 10 == 0:
            # record train things
            train_accuracy, train_cost, train_summary = sess.run([accuracy, cost, merge], feed_dict={
                x: batch_x,
                y: batch_y,
                # keep_prob: 0.5,
            })
            train_writer.add_summary(train_summary, step)

            # # record test things
            # test_accuracy, test_summary = sess.run([accuracy, merge],
            #                                        feed_dict={x: test_x,
            #                                                   y: test_y,
            #                                                   # keep_prob: 1.0,
            #                                                   })
            # test_writer.add_summary(test_summary, step)

            # # pring somethings
            # i_want_to_see(title='step '+str(step),
            #               content=[
            #     # 'confusion_matrix',
            #     # confusion_matrix.eval(),
            #     'lamda: '+str(lameda)+'   Learning rate: ' + \
            #     str(sess.run(learning_rate))+'    cost: '+str(train_cost),
            #     'train accuracy:    '+str(train_accuracy),
            #     'test accuracy :    '+str(test_accuracy),
            # ])

        # ############## early stopping ##################
        # if test_accuracy > 0.9999:
        #     print(
        #         "The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ", test_accuracy)
        #     break

    train_writer.close()
    test_writer.close()
