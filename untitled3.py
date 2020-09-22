# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:25:30 2020

@author: Administrator
"""

import time
import numpy as np
from keras.models import load_model
from keras.preprocessing import image




# 用训练好的模型来预测新的样本
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import load_model
import os
 
def predict(model, img_path, target_size):
    a=0
    b=0
#    model = load_model()
    fileList = os.listdir(img_path)

    for fileName in fileList: 
#        print(fileName)
        img_path_name=img_path+fileName
#        print(img_path_name)
        
        
#        img = cv2.imread(img_path_name)


#        img_height, img_width = 32, 32
        x = image.load_img(path=img_path_name, target_size=target_size)
        x = image.img_to_array(x)
        x = x[None]
    
    
        y = model.predict(x)
        print(y)
#        print(len(y))
#        print(int(y))
        k=int(y)
        if(k!=0):
            b=b+1
        else:
            a=a+1
    print(a)
    print(b)
    R=b/(a+b)
    return R


#    return '0'
 
if __name__ == '__main__':
#    model_path = '../cnn.hdf5'
    model = load_model('flowers.h5')
    target_size = (224, 224)
#    img_path = './data/test/nosmoke/'
#   img_path = r'./flowers/test/roses/'
    img_path = r'./flowers/train/sunflowers/'
    R=predict(model, img_path, target_size)
    print('识别准确率为%',R)