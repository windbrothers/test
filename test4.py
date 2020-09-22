# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:24:28 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:26:37 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:35:40 2019

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
可用
conda config --set show_channel_urls yes
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""#使用CPU
os.environ['KERAS_BACKEND'] ='tensorflow'#'theano'# 
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Flatten
from keras.layers.pooling import GlobalAveragePooling2D
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import keras
from keras.preprocessing.image import ImageDataGenerator
batch_size=128
datagen = ImageDataGenerator(rescale=1.0 / 255)
in_size=(32,32)
Train=True#False#
if Train:
    actv='relu'
   
    path='0'
    in_shape=(*in_size,3)

    base_num=2
    model = Sequential()
    
    model.add(Conv2D(8, (3, 3), input_shape=in_shape, padding='same', activation=actv))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3),  padding='same', activation=actv))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3),  padding='same', activation=actv))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    #model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Conv2D(128, (3, 3),  padding='same', activation=actv))
#    model.add(Conv2D(256, (3, 3),  padding='same', activation=actv))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3),  padding='same', activation=actv))
    model.add(Dropout(0.3))
    model.add(GlobalAveragePooling2D())#Flatten
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))#tf.nn.log_softmax

#    lrate = 0.001;decay = 0.007;momentum=0.3
    optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.007)
#    optimizer = optimizers.SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()#显示模型
    #import sys;sys.exit();
    #
    train = datagen.flow_from_directory('%s/Train'%path,target_size=in_size,batch_size=batch_size,class_mode='categorical',shuffle=False)
        
    from keras.callbacks import ModelCheckpoint,EarlyStopping,History
    history = History()
    model_checkpoint = ModelCheckpoint('cnn.hdf5', monitor='loss', save_best_only=True)
    EarlyStopping=EarlyStopping(monitor='acc', patience=50, verbose=2, mode='auto')
    callbacks = [
                history,
                model_checkpoint,
                EarlyStopping
            ]
    
    test = datagen.flow_from_directory('%s/test'%path,target_size=in_size,batch_size=batch_size,class_mode='categorical',shuffle=False)
    model.fit_generator(train,steps_per_epoch=len(train.classes)/train.batch_size,
            epochs=200,
            validation_data=test,
                  callbacks=callbacks,
            validation_steps=len(test.classes)/test.batch_size,
            verbose=2)
else:
    model=load_model('model-cnn1.h5')
    model.summary()
from keras.utils import np_utils
from sklearn import metrics
import numpy as np
import time
t0=time.time()
test = datagen.flow_from_directory('%s/test'%path,target_size=in_size,batch_size=batch_size,class_mode='categorical',shuffle=False)
y_pred_=model.predict_generator(test, len(test.classes)/test.batch_size)
test_labels=np_utils.to_categorical(test.classes)
y_true=test_labels.argmax(axis=1)
y_pred=y_pred_.argmax(axis=1)
print(y_true.shape,y_pred.shape)
#print(y_true,y_pred)
uniques = np.unique(y_true,axis=0)
print(uniques.shape, uniques)
classify_report = metrics.classification_report(y_true, y_pred)

confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
overall_accuracy = metrics.accuracy_score(y_true, y_pred)
acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
average_accuracy = np.mean(acc_for_each_class)
score = metrics.accuracy_score(y_true, y_pred)
print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
import pandas as pd
data1 = pd.DataFrame(confusion_matrix)
data1.to_csv('confusion_matrix.csv')

print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('score: {0:f}'.format(score))
acc_for_each_class=np.around(acc_for_each_class, decimals=2)
np.savetxt('afec_mycnn.csv',acc_for_each_class)

import matplotlib.pyplot as plt


plt.figure(12)
plt.subplot(121)
train_acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(len(train_acc))
plt.plot(epochs, train_acc, 'b',label='train_acc')
plt.plot(epochs, val_acc, 'r',label='test_acc')
plt.title('Train and Test accuracy')
plt.legend()
     
plt.subplot(122)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(train_loss))
plt.plot(epochs, train_loss, 'b',label='train_loss')
plt.plot(epochs, val_loss, 'r',label='test_loss')
plt.title('Train and Test loss')
plt.legend()
plt.show()