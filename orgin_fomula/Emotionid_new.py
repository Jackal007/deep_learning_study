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


def i_want_to_see(title='title', content=[]):
    '''
    print what you want to see in a cool way
    @param name:
    @param content:
    '''

    print('------------------------------ [',
          title, '] ------------------------------')
    for c in content:
        print(c)


def RNN(X):
    '''
    model structure
    @return predict result
    '''

    # neurons in hidden layer
    input_units = n_inputs
    fc1_in_size = input_units
    fc1_nuits = input_units
    fc2_nuits = input_units
    fc3_nuits = 256
    fc4_nuits = 256
    fc5_nuits = 256

    # Define weights and biases
    weights = {
        'lstm_out': tf.Variable(tf.random_normal([n_inputs, n_classes]), trainable=True),
        # 'input': tf.Variable(tf.random_normal([n_inputs, input_units]), trainable=True),
        # 'fc1': tf.Variable(tf.random_normal([fc1_in_size, fc1_nuits]), trainable=True),
        # 'fc2': tf.Variable(tf.random_normal([fc1_nuits, fc2_nuits]), trainable=True),
        # 'fc3': tf.Variable(tf.random_normal([fc2_nuits, fc3_nuits]), trainable=True),
        # 'fc4': tf.Variable(tf.random_normal([fc3_nuits, fc4_nuits]), trainable=True),
        # 'output': tf.Variable(tf.random_normal([fc1_nuits, n_classes]), trainable=True),
    }
    biases = {
        'lstm_out': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True),
        # 'input': tf.Variable(tf.constant(0.1, shape=[input_units]), trainable=True),
        # 'fc1': tf.Variable(tf.constant(0.1, shape=[fc1_nuits]), trainable=True),
        # 'fc2': tf.Variable(tf.constant(0.1, shape=[fc2_nuits]), trainable=True),
        # 'fc3': tf.Variable(tf.constant(0.1, shape=[fc3_nuits]), trainable=True),
        # 'fc4': tf.Variable(tf.constant(0.1, shape=[fc4_nuits]), trainable=True),
        # 'output': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True),
    }

    ###################################### model struct srart ############################################

    # 1------------------- fc -------------------
    # X (batch_size,n_step,input_units=n_inputs)===>X (?,input_units=n_inputs)
    # X = tf.reshape(X, [-1, input_units])

    # input_layer = tf.nn.relu(
    #     tf.add(
    #         tf.matmul(X, weights['input']),
    #         biases['input']))

    # # 2------------------- fc -------------------
    # # X (input_units, fc1_in_size)
    # fc1_layer = tf.nn.tanh(
    #     tf.add(
    #         tf.matmul(input_layer, weights['fc1']),
    #         biases['fc1']))

    # # 3------------------- fc -------------------
    # # X (input_units, fc1_in_size)
    # fc2_layer = tf.nn.sigmoid(
    #     tf.add(
    #         tf.matmul(fc1_layer, weights['fc2']),
    #         biases['fc2']))

    # fc_drop = tf.nn.dropout(fc2_layer, keep_prob)

    # 4------------------- lstm -------------------
    # X_hidd1 (?,input_units=n_inputs) ====> X_in (batch_size=?,n_step,input_units=n_inputs)
    # X_in = tf.reshape(X, [-1, n_step, n_inputs])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        n_inputs, forget_bias=1, state_is_tuple=True)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        lstm_cell, X, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    lstm_results = tf.matmul(outputs[-1], weights['lstm_out']) + \
        biases['lstm_out']  # 选取最后一个 output

    # fc3_layer = tf.nn.tanh(
    #     tf.add(
    #         tf.matmul(fc2_layer, weights['fc3']),
    #         biases['fc3']))
    # fc4_layer = tf.nn.sigmoid(
    #     tf.add(
    #         tf.matmul(fc3_layer, weights['fc4']),
    #         biases['fc4']))

    # results = tf.nn.relu(tf.matmul(fc1_layer, weights['output']) +
    #                      biases['output'])

    return lstm_results  # results  # , outputs_att


##################################################################### 跑模型 ##################################################################

# about data
n_step = get_n_step()
n_inputs = feature_number = get_feature_number()  # the size of input layer
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

#---------------------------------------------------------------

pred = RNN(x)

# L2 loss prevents this overkill neural network to overfit the data
l2 = lameda * sum(tf.nn.l2_loss(tf_var)
                  for tf_var in tf.trainable_variables())

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=pred, labels=y))+l2  # Softmax loss

train_op = tf.train.AdamOptimizer(
    learning_rate).minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# pred_result = tf.argmax(pred, 1, name="pred_result")
# label_true = tf.argmax(y, 1)
# confusion_matrix=tf.contrib.metrics.confusion_matrix(pred_result,label_true)

# tensorboard things
logfile = "log"
if os.path.exists(logfile):
    shutil.rmtree(logfile)

summary_loss = tf.summary.scalar("loss", cost)
summary_accuracy = tf.summary.scalar("accuracy", accuracy)

train_merge = tf.summary.merge([summary_loss, summary_accuracy])
test_merge = tf.summary.merge([summary_accuracy])

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logfile+"/train", sess.graph)
    test_writer = tf.summary.FileWriter(logfile+"/test", sess.graph)

    test_x, test_y = get_next_test_batch()

    for step in range(train_times):

        ################ train #################
        batch_x, batch_y = get_next_train_batch()
        sess.run([train_op], feed_dict={
            x: batch_x,
            y: batch_y,
            # keep_prob: 0.5,
        })
        sess.run(tf.local_variables_initializer())

        ############### print something ##################
        if step % 10 == 0:
            # record train things
            _, train_accuracy, train_cost, train_summary = sess.run([train_op, accuracy, cost, train_merge], feed_dict={
                x: batch_x,
                y: batch_y,
                # keep_prob: 0.5,
            })
            train_writer.add_summary(train_summary, step)

            # record test things
            test_accuracy, test_summary = sess.run([accuracy, test_merge],
                                                   feed_dict={x: test_x,
                                                              y: test_y,
                                                              # keep_prob: 1.0,
                                                              })
            test_writer.add_summary(test_summary, step)

            # pring somethings
            i_want_to_see(title='step '+str(step),
                          content=[
                # 'confusion_matrix',
                # confusion_matrix.eval(),
                'lamda: '+str(lameda)+'   Learning rate: ' + \
                str(sess.run(learning_rate))+'    cost: '+str(train_cost),
                'train accuracy:    '+str(train_accuracy),
                'test accuracy :    '+str(test_accuracy),
            ])

        # ############## early stopping ##################
        # if test_accuracy > 0.9999:
        #     print(
        #         "The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ", test_accuracy)
        #     break

    train_writer.close()
    test_writer.close()
