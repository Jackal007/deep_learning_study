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
        'input': tf.Variable(tf.random_normal([n_inputs, input_units]), trainable=True),
        'fc1': tf.Variable(tf.random_normal([fc1_in_size, fc1_nuits]), trainable=True),
        'fc2': tf.Variable(tf.random_normal([fc1_nuits, fc2_nuits]), trainable=True),
        'lstm_out': tf.Variable(tf.random_normal([fc2_nuits, n_classes]), trainable=True),
        # 'fc3': tf.Variable(tf.random_normal([fc2_nuits, fc3_nuits]), trainable=True),
        # 'fc4': tf.Variable(tf.random_normal([fc3_nuits, fc4_nuits]), trainable=True),
        # 'output': tf.Variable(tf.random_normal([fc1_nuits, n_classes]), trainable=True),
    }
    biases = {
        'input': tf.Variable(tf.constant(0.1, shape=[input_units]), trainable=True),
        'fc1': tf.Variable(tf.constant(0.1, shape=[fc1_nuits]), trainable=True),
        'fc2': tf.Variable(tf.constant(0.1, shape=[fc2_nuits]), trainable=True),
        'lstm_out': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True),
        # 'fc3': tf.Variable(tf.constant(0.1, shape=[fc3_nuits]), trainable=True),
        # 'fc4': tf.Variable(tf.constant(0.1, shape=[fc4_nuits]), trainable=True),
        # 'output': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True),
    }

    ###################################### model struct srart ############################################

    # 1------------------- fc -------------------
    # X (batch_size,n_step,input_units=n_inputs)===>X (?,input_units=n_inputs)
    X = tf.reshape(X, [-1, input_units])

    input_layer = tf.nn.relu(
        tf.add(
            tf.matmul(X, weights['input']),
            biases['input']))

    # 2------------------- fc -------------------
    # X (input_units, fc1_in_size)
    fc1_layer = tf.nn.relu(
        tf.add(
            tf.matmul(input_layer, weights['fc1']),
            biases['fc1']))

    # 3------------------- fc -------------------
    # X (input_units, fc1_in_size)
    fc2_layer = tf.nn.relu(
        tf.add(
            tf.matmul(fc1_layer, weights['fc2']),
            biases['fc2']))

    # 4------------------- lstm -------------------
    # X_hidd1 (?,input_units=n_inputs) ====> X_in (batch_size=?,n_step,input_units=n_inputs)
    X_in = tf.reshape(fc2_layer, [-1, n_step, input_units])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        input_units, forget_bias=1, state_is_tuple=True)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        lstm_cell, X_in, initial_state=init_state, time_major=False)

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
# tensorboard things
logfile = "log"
if os.path.exists(logfile):
    shutil.rmtree(logfile)

# 定义一些东西
n_step = get_n_step()

n_inputs = feature_number = get_feature_number()  # the size of input layer
batch_size = get_batch_size()


# 下面是和模型有关的
lameda = 0.001
train_times = 50000

learning_rate = 0.0001
# tf.train.exponential_decay(
#     learning_rate=0.01,
#     global_step=train_times,
#     decay_steps=100,
#     decay_rate=0.96,
#     staircase=True,
#     name=None
# )


n_classes = 3  # the size of output layer,there are 3 different kind of classes


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_step, feature_number], name="features")
y = tf.placeholder(tf.float32, [None, n_classes], name="labels")

pred = RNN(x)

# L2 loss prevents this overkill neural network to overfit the data
l2 = lameda * sum(tf.nn.l2_loss(tf_var)
                  for tf_var in tf.trainable_variables())
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=pred, labels=y))+l2  # Softmax loss


train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
pred_result = tf.argmax(pred, 1, name="pred_result")
label_true = tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# confusion_matrix=tf.contrib.metrics.confusion_matrix(pred_result,label_true)

summary_loss = tf.summary.scalar("train_loss", cost)
summary_accuracy = tf.summary.scalar("train_accuracy", accuracy)

streaming_loss, streaming_loss_update = tf.contrib.metrics.streaming_mean(cost)
streaming_loss_scalar = tf.summary.scalar('test_loss', streaming_loss)

streaming_accuracy, streaming_accuracy_update = tf.contrib.metrics.streaming_mean(
    accuracy)
streaming_accuracy_scalar = tf.summary.scalar(
    'test_accuracy', streaming_accuracy)

train_merge = tf.summary.merge([summary_loss, summary_accuracy])
test_merge = tf.summary.merge(
    [streaming_loss_scalar, streaming_accuracy_scalar])


init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
rnn_s = time.clock()
with tf.Session(config=config) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logfile+"/train", sess.graph)
    test_writer = tf.summary.FileWriter(logfile+"/test", sess.graph)

    lr = learning_rate  # sess.run(learning_rate)
    test_x, test_y = get_next_test_batch()

    for step in range(train_times):

        ################ train #################
        batch_x, batch_y = get_next_train_batch()

        # i_want_to_see('train_data',content=[batch_x,batch_y])

        _, train_accuracy, train_cost, summary = sess.run([train_op, accuracy, cost, train_merge], feed_dict={
            x: batch_x,
            y: batch_y,
        })

        train_writer.add_summary(summary, step)

        ############### print something ##################
        sess.run(tf.local_variables_initializer())

        if step % 20 == 0:

            test_accuracy, streaming_loss, streaming_accuracy = sess.run([accuracy, streaming_loss_update, streaming_accuracy_update],
                                                                         feed_dict={x: test_x,
                                                                                    y: test_y,
                                                                                    })
            summary = sess.run(test_merge)
            test_writer.add_summary(summary, step)

            i_want_to_see(title='step'+str(step),
                          content=[
                # 'confusion_matrix',
                # confusion_matrix.eval(),
                'lamda: '+str(lameda)+'   Learning rate: ' + \
                str(lr)+'    cost: '+str(train_cost),
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
