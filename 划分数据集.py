# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:45:37 2020

@author: Administrator
"""

import os
import random
import shutil
from shutil import copy2
path=os.getcwd()
trainfiles = os.listdir(path+'../yfn/set1/nosmoke')
num_train = len(trainfiles)
print( "num_train: " + str(num_train) )
index_list = list(range(num_train))
#print(index_list)
#print(trainfiles)
random.shuffle(index_list)
num = 0
#trainDir=r'D:\doshiyan\set1\train\nosmoke'
#testDir =r'D:\doshiyan\set1\test\nosmoke'
#for i in index_list:
#    fileName = os.path.join('D:\doshiyan\yfn\set1\smoke', trainfiles[i])
##    print(fileName)
#    if num < num_train*0.7:
#        print(str(fileName))
#        copy2(fileName, trainDir)
#    else:
#        copy2(fileName, testDir)
#    num += 1