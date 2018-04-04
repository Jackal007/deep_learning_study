'''
Created on 2018年1月31日

@author: zheng
'''

import numpy as np

from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Flatten
from keras.utils import to_categorical
from keras.applications import ResNet50
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import Adam
from utils import Img_Data_Processer

class My_Model():
    
    def __init__(self):
        self.model = self.get_model()
    
    def get_dataset(self):
        from utils import Img_Data_Processer
        t=Img_Data_Processer()
        _,self.train_X ,self.train_Y = t.get_titles_and_Xs_and_Ys('train')
        _,self.test_X ,self.test_Y = t.get_titles_and_Xs_and_Ys('test')
        self.validation_title,self.validation_X ,_ = t.get_titles_and_Xs_and_Ys('validation')
    
    def get_model(self):
        img_height,img_width=Img_Data_Processer().img_height,Img_Data_Processer().img_width
        model = ResNet50(include_top=False, input_shape=[img_height,img_width, 3])
        model.summary()
        model.trainable = True
        x = model.output
        out = Flatten()(x)
        out = Dense(units=1024, activation='relu', kernel_regularizer=l2(0.001))(out)
        out = Dense(units=512, activation='relu', kernel_regularizer=l2(0.001))(out)
        out = Dense(units=26,activation='softmax', kernel_regularizer=l2(0.001))(out)
        new_model = Model(inputs=model.input, outputs=out)
        
        return new_model
    
    def train_model(self,train_times=500):
        model = self.model
        ten=10#need a modification
        img_height,img_width=Img_Data_Processer().img_height,Img_Data_Processer().img_width
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        for _ in range(train_times):
            selected = np.random.randint(0, len(self.train_X),[ten])
            X,Y = np.zeros([ten, img_height, img_width, 3]),np.zeros([ten, 26])
            for i in range(0,ten):
                X[i,...]=self.train_X[selected[i]]
                Y[i,...]=self.train_Y[selected[i]]
                
            model.fit(X, Y, batch_size=50, epochs=10, shuffle=True)
    
    def save_model(self):
        self.model.save('model.h5')
    
    def get_predict_result(self, X):
        model = self.model
        result = model.predict(X)
        strr='abcdefghijklmnopqrstuvwxyz'
        for c in strr:
            flag=True
            for i in range(0,len(result)-1):
                if result[i] != to_categorical(c,26)[i]:
                    flag=False
                    break
            if flag:
                result=c
                break
        
        return result
    
    def get_accuracy(self):
        from utils import gxl_Data_Processer
        
        titles,Xs=self.validation_title,self.validation_X
        predict_results=[]
        for X in Xs:
            t=np.zeros([1,210,280,3])
            t[0,...]=X
            predict_result=self.get_predict_result(t)
            predict_results.append(predict_result)
        
        _,_,validaion_datas=gxl_Data_Processer().get_which_to_train_test_validation()
        
        right,wrong=0,0
        for title,predict_result in zip(titles,predict_results):
            print(predict_result)
            if title in validaion_datas[predict_result] :
                right+=1
            else:
                wrong+=1
        
        return (right*1.0)/((right+wrong)*1.0)
    
def main():
    t=My_Model()
    t.get_dataset()
    t.get_model()
    t.train_model()
    print(t.get_accuracy())
    
if __name__=='__main__':
    from utils import gxl_Data_Processer
    t = gxl_Data_Processer()
    t.get_which_to_train_test_validation()
    t.gxls_to_imgs()
    main()