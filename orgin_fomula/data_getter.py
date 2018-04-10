'''
Created on 2018年4月5日

@author: Jack
'''
import os
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
from scipy.signal import butter, lfilter

batch_size = 4096
n_step = 1  # 因为数据等下要交给lstm来处理
overlap = 2
feature_number = 62*overlap
train_dataset_dir = '../seed_data/train/'
test_dataset_dir = '../seed_data/test/'

train_x = []
train_y = []
train_datas_len = 0
train_datas_cursor = 0

test_x = []
test_y = []
test_datas_len = 0


labels = [1, 0, - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 0, 1, - 1]


def one_hot(y_, values=[]):
    '''
    # this function is used to transfer one column label to one hot label
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    @param values:different values
    @return: one_hot_code
    '''
    y_ = np.array(y_).astype(int)
    n_values = len(values)

    # 因为onehot不能处理负数，所以把所有的数加上最小的负数的绝对值
    if min(values) < 0:
        y_ = y_-min(values)

    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    extract the delta from the data
    @return: b,a
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    @return: y
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def data_preprocess(xs, ys):
    '''
    @return processed data
    '''
    processed_xs = np.array(xs).reshape(-1,  feature_number)

    # # filter the wave
    temp = np.array(butter_bandpass_filter(
        data=processed_xs[:, 0], lowcut=12, highcut=30, fs=200, order=3)).reshape(-1, 1)
    for i in range(1, processed_xs.shape[1]):
        filtered_wave = np.array(butter_bandpass_filter(
            data=processed_xs[:, i], lowcut=12, highcut=30, fs=200, order=3)).reshape(-1, 1)
        temp = np.hstack((temp, filtered_wave))

    # 归一化
    processed_xs = preprocessing.scale(temp)
    processed_xs = processed_xs.reshape(-1, n_step, feature_number)

    # onehot化
    processed_ys = one_hot(ys, values=[-1, 0, 1])

    return processed_xs, processed_ys


def get_train_datas():
    '''
    get random datas from train dataset
    @return: train_x,train_y
    '''
    global train_x, train_y
    global train_datas_len

    train_x = []
    train_y = []

    record_list = [task for task in os.listdir(
        train_dataset_dir) if os.path.isfile(os.path.join(train_dataset_dir, task))]
    record_name = record_list[np.random.randint(0, len(record_list))]

    record = sio.loadmat(train_dataset_dir+"/"+record_name)
    data_keys = [key for key in record.keys() if '1' in key]

    for eeg_num in data_keys:
        student_data = record[eeg_num].transpose(1, 0)
        student_data_len = len(student_data)
        y = labels[int(eeg_num) - 101]
        cursor = 0
        while cursor+n_step*overlap < student_data_len:
            x = student_data[cursor:cursor+n_step*overlap]
            cursor += n_step

            if np.array(x).shape[0] != n_step*overlap:
                continue

            train_x.append(x.reshape(feature_number))
            train_y.append(y)

    train_x, train_y = data_preprocess(train_x, train_y)
    train_datas_len = len(train_y)


def get_test_datas():
    '''
    get all datas from test dataset
    @return: test_x,test_y
    '''

    global test_x, test_y
    global test_datas_len

    record_list = [task for task in os.listdir(
        test_dataset_dir) if os.path.isfile(os.path.join(test_dataset_dir, task))]
    for record_name in record_list:
        record = sio.loadmat(test_dataset_dir+"/"+record_name)
        for eeg_num in record.keys():
            if '1' in eeg_num:
                student_data = record[eeg_num].transpose(1, 0)
                student_data_len = len(student_data)
                y = labels[int(eeg_num) - 101]
                cursor = 0
                while cursor+n_step*overlap < student_data_len:
                    x = student_data[cursor:cursor+n_step*overlap]
                    cursor += n_step

                    if np.array(x).shape[0] != n_step*overlap:
                        continue

                    test_x.append(x.reshape(feature_number))
                    test_y.append(y)

    test_x, test_y = data_preprocess(test_x, test_y)
    test_datas_len = len(test_y)


def get_next_train_batch():
    '''
    @return train_x and train_y in batch size
    '''
    global train_datas_cursor

    if train_datas_cursor+batch_size >= train_datas_len:
        train_datas_cursor = 0
        get_train_datas()

    x = train_x[train_datas_cursor:train_datas_cursor+batch_size]
    y = train_y[train_datas_cursor:train_datas_cursor+batch_size]

    train_datas_cursor += batch_size

    return x, y


def get_next_test_batch():
    '''
    @return test_x and test_y in batch size
    '''

    if test_datas_len <= 0:
        get_test_datas()

    x = test_x[0:batch_size]
    y = test_y[0:batch_size]

    # x = []
    # y = []
    # while len(x) < batch_size:
    #     random_index = np.random.randint(0, test_datas_len-1)
    #     x.append(test_x[random_index])
    #     y.append(test_y[random_index])

    return x, y


def get_feature_number():

    return feature_number


def get_batch_size():

    return batch_size


def get_n_step():

    return n_step
