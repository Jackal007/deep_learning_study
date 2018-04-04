'''
Created on 2018年1月31日

@author: zheng
'''

import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__ == '__main__':
    from utils import Img_Data_Processer,gxl_Data_Processer
    from my_model import My_Model
    t=Img_Data_Processer()
    X=np.zeros([1,210,280,3])
    _,X[0,...]=t.get_a_img_title_and_X('imgs/test/Z-0.png')
    t=gxl_Data_Processer()
    t.get_which_to_train_test_validation()
    #t.gxls_to_imgs()
    tt=My_Model()
    tt.get_dataset()
    tt.get_model()
    tt.train_model(train_times=1)
    tt.save_model()
    result = tt.get_predict_result(X)
    
    print(result)
    

