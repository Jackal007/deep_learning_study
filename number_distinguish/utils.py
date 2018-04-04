'''
Created on 2018年1月31日
  
@author: zheng
'''
  
import os
import cv2
import numpy as np
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pylab as pl
import xml.dom.minidom
 
class gxl_Data_Processer():
     
    def __init__(self):
        self.train_datas = dict()
        self.test_datas = dict()
        self.validation_datas = dict()
     
    def get_which_to_train_test_validation(self):
        '''
        get who to be used as train,test,validation datasets
        '''
        train_file = 'train.cxl';test_file = 'test.cxl';validation_file = 'validation.cxl'
        records = []
        for file_name in [train_file, test_file, validation_file]:
            records.append(self.get_gxl_names_and_labels(record_file='gxls/' + file_name))
        self.train_datas = records[0]
        self.test_datas = records[1]
        self.validation_datas = records[2]
        return self.train_datas,self.test_datas,self.validation_datas
         
    def get_gxl_names_and_labels(self, record_file):
        '''
        get gxl name and class by usage['train','test','validation']
        '''
        record = dict()
        root = xml.dom.minidom.parse(record_file).documentElement
        prints = root.getElementsByTagName('print')
        for oneprint in prints:
            file_name = oneprint.getAttribute('file')
            label = oneprint.getAttribute('class').lower()
            try:
                record[label].append(file_name)
            except:
                record[label] = [file_name, ]
         
        return record
         
    def gxls_to_imgs(self):
        for k in self.train_datas.keys():
            for i in range(0, len(self.train_datas[k]) - 1):
                self.a_gxl_to_img('gxls/' + self.train_datas[k][i], 'imgs/train/' + k + '-' + str(i),usage='train')
        for k in self.test_datas.keys():
            for i in range(0, len(self.test_datas[k]) - 1):
                self.a_gxl_to_img('gxls/' + self.test_datas[k][i], 'imgs/test/' + k + '-' + str(i),usage='test')
        for k in self.validation_datas.keys():
            for i in range(0, len(self.validation_datas[k]) - 1):
                self.a_gxl_to_img('gxls/' + self.validation_datas[k][i], 'imgs/validation/' + k + '-' + str(i),usage='validation')
            
    
    def a_gxl_to_img(self, gxl_path, save_path,usage='train'):
        from PIL import Image
        dom = xml.dom.minidom.parse(gxl_path).documentElement
        
        # get Points
        my_nodes = dom.getElementsByTagName('node')
        Xs, Ys = [], []
        for my_node in my_nodes:
            Xs.append(float(my_node.childNodes[0].childNodes[0].childNodes[0].nodeValue))
            Ys.append(float(my_node.childNodes[1].childNodes[0].childNodes[0].nodeValue))
        
        # get Lines
        my_edges = dom.getElementsByTagName('edge')
        for my_edge in my_edges:
            line_start = int(my_edge.getAttribute('from')[1:])
            line_end = int(my_edge.getAttribute('to')[1:])
            plt.plot([Xs[line_start], Xs[line_end]], [Ys[line_start], Ys[line_end]])
        
        plt.title("")
        plt.xlim(xmax=10, xmin=0);plt.ylim(ymax=10, ymin=0);
        plt.figure(figsize=(Img_Data_Processer.img_width/100, Img_Data_Processer.img_height/100))
        #plt.xticks([]);plt.yticks([]);
        plt.plot(Xs, Ys)
        plt.axis('off')
        plt.savefig(save_path)
        if usage=='train':
            for i in [45,90,135,180,225,270,315]:
                image=Image.open(save_path+'.png')
                image = image.rotate(int(i))
                plt.imshow(image)
                plt.axis('off')
                path=save_path+'-'+str(i)
                plt.savefig(path)
        
        plt.close()
  
 
class Img_Data_Processer():
    
    img_width = 280
    img_height = 210
      
    def get_a_img_title_and_X(self, img_path):
        '''
        get a img data
        '''
        img = cv2.imread(img_path)
        title=img_path.split('/')[-1]
        return title,img
      
    def get_a_img_Y(self, img_path):
        '''
        get a img label
        '''
        from keras.utils import to_categorical
        return to_categorical(ord(img_path.lower()[0])-ord('a'), 26)
      
    def get_titles_and_Xs_and_Ys(self, path='train'):
        '''
        get Xs
        for
        train test or validation
        '''
        path = 'imgs/' + path + '/'
        
        imgs_paths = os.listdir(path)
        titles=[]
        Xs = np.zeros([len(imgs_paths), self.img_height, self.img_width, 3])
        Ys = np.zeros([len(imgs_paths), 26])
        for index, img_path in zip(range(0, len(imgs_paths) - 1), imgs_paths):
            title,Xs[index, ...] = self.get_a_img_title_and_X(path + img_path)
            titles.append(title)
            Ys[index, ...] = self.get_a_img_Y(img_path)
                 
        return titles, Xs, Ys
    
def main():
    t = gxl_Data_Processer()
    t.get_which_to_train_test_validation()
    t.gxls_to_imgs()
    #t = Img_Data_Processer()
    #t.get_titles_and_Xs_and_Ys(path='validation')
    
if __name__ == '__main__':
    main()
