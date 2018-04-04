import os
import time
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np

np.random.seed(33)

# input data paramters
n_channel = 200
input_height = 9
input_width = 9

n_labels = 3

n_lstm_layers = 2

dropout_prob = 0.5

calibration = 'N'
norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = False

output_dir = "a"
output_file = "b"


def i_want_to_see(name, content):
    '''
    print what you want to see in a cool way
    @param name:
    @param content:
    '''

    print('#####################################################################################')
    print(time.asctime(time.localtime(time.time())))
    print(name)
    print('-------------------------------------------------------------------------------------')
    print(content)
    print('#####################################################################################')


def one_hot(y_):
    '''
    # this function is used to transfer one column label to one hot label
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    @return: one_hot_code
    '''
    y_ = y_.reshape(len(y_))
    y_ = y_.astype(int) + 1
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def get_x_y(train_or_test='train', dataset_dir='../output/'):
    '''
    @param train_or_test:get train x,y or get test x,y
    @param dataset_dir:the directory store the datas
    @return : x, y
    '''
    if train_or_test == 'train':
        with open(dataset_dir+"_data_200.pkl", "rb") as fp:
            features = pickle.load(fp)
        with open(dataset_dir+"_label_200.pkl", "rb") as fp:
            labels = pickle.load(fp)

    elif train_or_test == 'test':
        with open(dataset_dir+"_data_test_200.pkl", "rb") as fp:
            features = pickle.load(fp)
        with open(dataset_dir+"_label_test_200.pkl", "rb") as fp:
            labels = pickle.load(fp)

    else:
        raise Exception('wrong choice for train_or_test')

    features = features.reshape(-1, input_height, input_width, n_channel)
    labels = one_hot(labels)

    return features, labels


def weight_variable(shape):
    '''
    @param
    @return
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    '''
    @param
    @return
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')


def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride):
    weight = weight_variable(
        [filter_height, filter_width, in_channels, out_channels])
    # each feature map shares the same weight and bias
    bias = bias_variable([out_channels])
    return tf.nn.elu(tf.add(conv2d(x, weight, kernel_stride), bias))


def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')


def apply_fully_connect(x, x_size, fc_size):
    fc_weight = weight_variable([x_size, fc_size])
    fc_bias = bias_variable([fc_size])
    return tf.nn.elu(tf.add(tf.matmul(x, fc_weight), fc_bias))


def apply_readout(x, x_size, readout_size):
    readout_weight = weight_variable([x_size, readout_size])
    readout_bias = bias_variable([readout_size])
    return tf.add(tf.matmul(x, readout_weight), readout_bias)


def model_things(train_x, train_y, test_x, test_y):
    '''
    should be gave a better name
    @return: test_pred_1_hot, test_true_list
    '''

    # training parameter
    lambda_loss_amount = 0.0005

    batch_size = 100

    training_epochs = (train_x.shape[0]//batch_size)

    # algorithn parameter
    learning_rate = 1e-4
    # learning_rate = 0.001

    # input placeholder
    X = tf.placeholder(tf.float32,
                       shape=[None, input_height,
                              input_width, n_channel],
                       name='X')
    Y = tf.placeholder(tf.float32,
                       shape=[None, n_labels],
                       name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    ####################################################### model struct start ######################################################

    # conv parameter
    conv_1_shape = '3*3*1*32'
    pool_1_shape = '2*2'

    conv_2_shape = '3*3*1*64'
    pool_2_shape = '2*2'

    conv_3_shape = '3*3*1*128'

    # full connected parameter
    fc_size = 1024
    n_fc_in = 1024
    n_fc_out = 1024

    # kernel parameter
    kernel_height_1st = 3
    kernel_width_1st = 3

    kernel_height_2nd = 3
    kernel_width_2nd = 3

    kernel_height_3rd = 3
    kernel_width_3rd = 3

    kernel_stride = 1
    conv_channel_num = 32

    # pooling parameter
    pooling_height = 2
    pooling_width = 2

    pooling_stride = 2
    ################ first CNN layer ################
    conv_1 = apply_conv2d(X,
                          kernel_height_1st, kernel_width_1st, n_channel, conv_channel_num,
                          kernel_stride)
    pool_1 = apply_max_pooling(conv_1,
                               pooling_height, pooling_width,
                               pooling_stride)

    ################ second CNN layer ################
    conv_2 = apply_conv2d(pool_1,
                          kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num*2,
                          kernel_stride)
    pool_2 = apply_max_pooling(conv_2,
                               pooling_height, pooling_width,
                               pooling_stride)

    ################ third CNN layer ################
    conv_3 = apply_conv2d(pool_2,
                          kernel_height_3rd, kernel_width_3rd,
                          conv_channel_num*2, conv_channel_num*4,
                          kernel_stride)

    ################ fully connected layer ################

    shape = conv_3.get_shape().as_list()
    conv_3_flat = tf.reshape(conv_3, [-1, shape[1]*shape[2]*shape[3]])

    fc = apply_fully_connect(conv_3_flat,
                             shape[1]*shape[2]*shape[3], fc_size*2)

    ################ dropout regularizer ################
    # Dropout (to reduce overfitting; useful when training very large neural network)
    # We will turn on dropout during training & turn off during testing
    fc_drop = tf.nn.dropout(fc, keep_prob)

    #####################################################
    # add lstm cell to network
    #####################################################
    # fc_drop size [batch_size*n_channel, fc_size]
    # lstm_in size [batch_size, n_channel, fc_size]
    lstm_in = tf.reshape(fc_drop, [-1, n_channel, fc_size])

    # define lstm cell
    # cells = []
    # for _ in range(n_lstm_layers):
    #     cells.append(
    #         tf.contrib.rnn.BasicLSTMCell(n_fc_in, forget_bias=1.0, state_is_tuple=True))
    # cells.append(
    #     tf.contrib.rnn.LSTMBlockCell(n_fc_in, forget_bias=1.0))
    # cells.append(
    #     tf.contrib.rnn.GRUBlockCell(n_fc_in, forget_bias=1.0, state_is_tuple=True))
    # cells.append(
    #     tf.contrib.rnn.GridLSTMCell(n_fc_in, forget_bias=1.0, state_is_tuple=True))
    # cells.append(
    #     tf.contrib.rnn.GLSTMCell(n_fc_in, forget_bias=1.0, state_is_tuple=True))
    # cell = tf.contrib.rnn.GRUCell(n_fc_in, state_is_tuple=True)
    # cells.append(cell)
    # cells.append(
    #     tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob))

    lstm_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.BasicLSTMCell(
            n_fc_in, forget_bias=1.0, state_is_tuple=True)
    ])

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # output ==> [batch, step, n_fc_in]
    output, states = tf.nn.dynamic_rnn(
        lstm_cell, lstm_in, initial_state=init_state, time_major=False)
    ################ output layer ################

    # output ==> [step, batch, n_fc_in]
    # output = tf.transpose(output, [1, 0, 2])

    # only need the output of last time step
    # rnn_output ==> [batch, n_fc_in]
    # rnn_output = tf.gather(output, int(output.get_shape()[0])-1)
    ###################################################################
    # another output method
    output = tf.unstack(tf.transpose(output, [1, 0, 2]), name='lstm_out')
    rnn_output = output[-1]
    ###################################################################

    ###########################################################################################
    # fully connected and readout
    ###########################################################################################
    # rnn_output ==> [batch, fc_size]
    shape_rnn_out = rnn_output.get_shape().as_list()
    # fc_out ==> [batch_size, n_fc_out]
    fc_out = apply_fully_connect(rnn_output, shape_rnn_out[1], n_fc_out)

    # keep_prob = tf.placeholder(tf.float32)
    fc_drop = tf.nn.dropout(fc_out, keep_prob)

    # readout layer
    y_ = apply_readout(fc_drop, shape_rnn_out[1], n_labels)
    y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
    y_posi = tf.nn.softmax(y_, name="y_posi")

    # l2 regularization
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )

    ####################################################### model struct end ######################################################

    ################ define loss ################
    if enable_penalty:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_, labels=Y) + l2, name='loss')
    else:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_, labels=Y), name='loss')

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # get correctly predicted object and accuracy
    correct_prediction = tf.equal(
        tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32), name='accuracy')

    ####################################################### run model ######################################################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())

        batch_start_index = np.random.randint(train_x.shape[0]-batch_size-1)
        batch_x = train_x[batch_start_index:batch_start_index+batch_size]
        batch_y = train_y[batch_start_index:batch_start_index+batch_size]

        for _ in range(training_epochs):
            _, c = session.run([train_op, cost], feed_dict={
                X: batch_x, Y: batch_y, keep_prob: 1-dropout_prob, phase_train: True})

        # save model
        saver = tf.train.Saver()
        saver.save(session, "./result/"+output_dir+"/model_"+output_file)


if __name__ == '__main__':
    # get data
    train_x, train_y = get_x_y(
        train_or_test='train', dataset_dir='../output/')
    test_x, test_y = get_x_y(
        train_or_test='test', dataset_dir='../output/')

    # do model things
    model_things(train_x, train_y, test_x, test_y)
