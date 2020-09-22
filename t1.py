# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:14:07 2020

@author: Administrator
"""

#深度学习006-移花接木-用Keras迁移学习提升性能 https://www.jianshu.com/p/4e3d13a7f4bf
#此处我们用VGG16的卷积层统一提取所有图片的特征，将这些特征保存，然后直接加载特征来训练，加载数字比加载图片要快的多，故而训练也快得多
#可以看出，主要调整Conv block 5，前面的4个block不需调整。  https://www.jianshu.com/p/5c766be9a9d7
import numpy as np
import os,random,shutil
np.random.seed(7)
from keras.applications.resnet50 import ResNet50

# 1, 准备数据集
#1，指定一些超参数：
train_data_dir='../data/data2/train'  # 训练集目录/train'
val_data_dir='../data/data2/test' # keras中将测试集称为validation set
train_samples_num=5350 # train set中全部照片数
val_samples_num=2300
IMG_W,IMG_H,IMG_CH=50,50,3 # 单张图片的大小
batch_size=50 # 不能是32，因为2000/32不能整除，后面会有影响。
epochs=180  # 用比较少的epochs数目做演示，节约训练时间



# 此处的训练集和测试集并不是原始图片的train set和test set，而是用VGG16对图片提取的特征，这些特征组成新的train set和test set
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255) # 不需图片增强  #生成一个 生成器

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    model = ResNet50(include_top=False, weights='imagenet')
    # 使用imagenet的weights作为VGG16的初始weights,由于只是特征提取，故而只取前面的卷积层而不需要DenseLayer，故而include_top=False

    generator = datagen.flow_from_directory( # 产生train set  以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
        train_data_dir,  #train数据集扩充
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='categorical', #shuffle函数的是将序列中的所有元素随机排序
        shuffle=False) # 必须为False，否则顺序打乱之后，和后面的label对应不上。
    bottleneck_features_train = model.predict_generator(  #预训练
        generator, train_samples_num // batch_size) # 如果是32，这个除法得到的是62，抛弃了小数，故而得到1984个sample
    np.save('./npy/VGG19_mydata_features_train.npy', bottleneck_features_train)
    print('features of train set is saved.')

    generator = datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, val_samples_num // batch_size)  #这个验证 相当于 test数据集
    np.save('./npy/VGG19_mydata_features_val.npy',bottleneck_features_validation)
    print(' features of test set is saved.')
#save_bottlebeck_features()
#上述通过扩充train和验证（test）数据集，然后在模型中预训练 提取特征 include_top=False保证了顶端的分类层失活
#
def my_model():
    '''
    自定义一个模型，该·模型仅仅相当于一个分类器，只包含有全连接层，对提取的特征进行分类即可
    :return:
    '''
    # 模型的结构
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:])) # 将所有data进行flatten
    model.add(Dense(256, activation='relu')) # 256个全连接单元
    model.add(Dropout(0.5)) # dropout正则
#    model.add(Dense(2, activation='softmax')) # 与二分类不同之处：要用Dense(class_num)和softmax
    
#    model = Sequential()
#    model.add(Flatten(input_shape=train_data.shape[1:])) # 将所有data进行flatten  train_data：训练图片集的所有图片
#    model.add(Dense(256, activation='relu')) # 256个全连接单元
#    model.add(Dropout(0.5)) # dropout正则
#    model.add(Dense(1, activation='sigmoid')) # 此处定义的模型只有后面的全连接层，由于是本项目特殊的，故而需要自定义  2分类的模型结构
##model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))#tf.nn.log_softmax
    # 模型的配置
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['acc']) # model的optimizer等
    model.summary()
    return model


from keras.utils import to_categorical
# 只需要训练分类器模型即可，不需要训练特征提取器
train_data = np.load('./npy/VGG19_mydata_features_train.npy') # 加载训练图片集的所有图片的VGG16-notop特征
train_labels = np.array(
    [0] * int((train_samples_num / 2)) + [1] * int((train_samples_num / 2)))
# label是1000个cat，1000个dog，由于此处VGG16特征提取时是按照顺序，故而[0]表示cat，1表示dog
train_labels = to_categorical(train_labels) 
print(train_labels)
validation_data = np.load('./npy/VGG19_mydata_features_val.npy')
validation_labels = np.array(
    [0] * int((val_samples_num / 2)) + [1] * int((val_samples_num / 2)))
validation_labels = to_categorical(validation_labels) 
print(validation_labels)
##label是400个cat，400个dog，由于此处VGG16特征提取时是按照顺序，故而[0]表示cat，1表示dog
# 构建分类器模型
clf_model=my_model()

history_ft = clf_model.fit(train_data,
              train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

import matplotlib.pyplot as plt
def plot_training(history):
    plt.figure(12)

    plt.subplot(121) #第一幅子图
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, 'b', label='train_acc')
    plt.plot(epochs, val_acc, 'r', label='test_acc')
    plt.title('Train and Test accuracy')
    plt.legend()

    plt.subplot(122)  #第二幅子图
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, 'b', label='train_loss')
    plt.plot(epochs, val_loss, 'r', label='test_loss')
    plt.title('Train and Test loss')
    plt.legend()

    plt.show()


plot_training(history_ft)
# 将本模型保存一下
clf_model.save_weights('./mydata_smoke_model_weight')
