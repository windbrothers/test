# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:18:44 2020

@author: Administrator
"""

# 用训练好的模型来预测新的样本
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import load_model
import os
 
def predict(model, img_path, target_size):
    a=0
    b=0
#    s=[]
    fileList = os.listdir(img_path)

    for fileName in fileList: 
#        print(fileName)
        img_path_name=img_path+fileName
#        print(img_path_name)
        img = cv2.imread(img_path_name)
    #    img=cv2.resize(32,24)
        if img.shape != target_size:
            print(fileName)
            print(img.shape)
            img = cv2.resize(img, target_size)
            print(img.shape)
            
        x = image.img_to_array(img)
        x *= 1. / 255  # 相当于ImageDataGenerator(rescale=1. / 255)
        x = np.expand_dims(x, axis=0)  # 调整图片维度
        preds = model.predict(x)
        print(preds)
#        if(preds[0][0]>preds[0][1]):
#            a=a+1
#            print('s')
#            path_S='check/s/'+fileName
##            print(path_S)
##            img = cv2.resize(img, (4000,3000))
#            cv2.imwrite(path_S,img)
#        else:
#            print('no')
#            path_S='check/no/'+fileName
#            print(path_S)
##            img = cv2.resize(img, (4000,3000))
##            cv2.imwrite(path_S,img)
#            b=b+1
#    return a,b
#    return '0'
 
if __name__ == '__main__':
    model_path = 'dogcatmodel.h5'
    model = load_model(model_path)
    target_size = (256, 256)
    img_path = './flowers/train/roses/'
#    img_path = '0/test/nosmoke/'
    a,b = predict(model, img_path, target_size)
    print(a,b)
    x=b/a
    print(x)
#    print(res)