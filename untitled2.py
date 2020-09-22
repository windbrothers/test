# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 20:54:39 2020

@author: Administrator
"""

from keras import optimizers
from keras import applications
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# 数据集
img_height, img_width = 32, 32  # 图片高宽
batch_size = 2  # 批量大小
epochs = 50  # 迭代次数
#train_data_dir = './data/train'  # 训练集目录
#validation_data_dir = './data/test'  # 测试集目录

train_data_dir = './flowers/train'  # 训练集目录
validation_data_dir = './flowers/test'  # 测试集目录
OUT_CATEGORIES = 1  # 分类数
nb_train_samples = 500  # 训练样本数
nb_validation_samples = 483  # 验证样本数

# 定义模型
base_model = applications.VGG16(weights="imagenet", include_top=False,
                                input_shape=(img_width, img_height, 3))  # 预训练的VGG16网络，替换掉顶部网络
print(base_model.summary())

for layer in base_model.layers[:15]: layer.trainable = False  # 冻结预训练网络前15层

top_model = Sequential()  # 自定义顶层网络
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  # 将预训练网络展平
top_model.add(Dense(256, activation='relu'))  # 全连接层，输入像素256
top_model.add(Dropout(0.5))  # Dropout概率0.5
top_model.add(Dense(1, activation='softmax'))#tf.nn.log_softmax
#top_model.add(Dense(OUT_CATEGORIES, activation='sigmoid'))  # 输出层，二分类

print(top_model.summary())
# top_model.load_weights("")  # 单独训练的自定义网络

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))  # 新网络=预训练网络+自定义网络

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])  # 损失函数为二进制交叉熵，优化器为SGD


train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)  # 训练数据预处理器，随机水平翻转
test_datagen = ImageDataGenerator(rescale=1. / 255)  # 测试数据预处理器
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='binary')  # 训练数据生成器
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width),
                                                        batch_size=batch_size, class_mode='binary',
                                                        shuffle=False)  # 验证数据生成器
checkpointer = ModelCheckpoint(filepath='smokemodel.h5', verbose=1, save_best_only=True)  # 保存最优模型
   


# 训练&评估
model.fit_generator(train_generator, 
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator, 
                    validation_steps=nb_validation_samples // batch_size,
                    verbose=2, workers=12, callbacks=[checkpointer])  # 每轮一行输出结果，最大进程12