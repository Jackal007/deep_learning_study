import scipy.io as sio
import os

labels = [1, 0, - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 0, 1, - 1]
label_in = sio.loadmat("../label.mat")


def get_data(data_dir='../seed_data/'):

    data = []

    record_list = [task for task in os.listdir(
        data_dir) if os.path.isfile(os.path.join(data_dir, task))]
    for record in record_list:
        student_datas = sio.loadmat(data_dir+"/"+record)
        for eeg_num in student_datas.keys():
            try:
                eeg_num = int(eeg_num)
                student_data = student_datas[str(eeg_num)]
                eeg_num -= 110
                for d in student_data:
                    data.append([d, labels[eeg_num]])
            except:
                pass

    return data


print(get_data())
