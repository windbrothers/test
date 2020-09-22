from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
#from keras.applications.mobilenet import preprocess_input

# 训练和测试的图片分为'bus', 'dinosaur', 'flower', 'horse', 'elephant'五类
# 其图片的下载地址为 http://pan.baidu.com/s/1nuqlTnN ,总共500张图片,其中图片以3,4,5,6,7开头进行按类区分
# 训练图片400张，测试图片100张；注意下载后，在train和test目录下分别建立上述的五类子目录，keras会按照子目录进行分类识别
NUM_CLASSES = 5
TRAIN_PATH = './flower_photos/train'
TEST_PATH = './flower_photos/test'
# 代码最后挑出一张图片进行预测识别
PREDICT_IMG = './flower_photos/test/rose/rose566.jpg'
# FC层定义输入层的大小
FC_NUMS = 1024
# 冻结训练的层数，根据模型的不同，层数也不一样，根据调试的结果，VGG19和VGG16c层比较符合理想的测试结果，本文采用VGG19做示例
FREEZE_LAYERS = 17
# 进行训练和测试的图片大小，VGG19推荐为224×244
IMAGE_SIZE = 224

# 采用VGG19为基本模型，include_top为False，表示FC层是可自定义的，抛弃模型中的FC层；该模型会在~/.keras/models下载基本模型
base_model = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# 自定义FC层以基本模型的输入为卷积层的最后一层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(FC_NUMS, activation='relu')(x)
prediction = Dense(NUM_CLASSES, activation='softmax')(x)

# 构造完新的FC层，加入custom层
model = Model(inputs=base_model.input, outputs=prediction)
# 可观察模型结构
model.summary()
# 获取模型的层数
print("layer nums:", len(model.layers))


# 除了FC层，靠近FC层的一部分卷积层可参与参数训练，
# 一般来说，模型结构已经标明一个卷积块包含的层数，
# 在这里我们选择FREEZE_LAYERS为17，表示最后一个卷积块和FC层要参与参数训练
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True
for layer in model.layers:
    print("layer.trainable:", layer.trainable)

# 预编译模型
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# 给出训练图片的生成器， 其中classes定义后，可让model按照这个顺序进行识别
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(directory=TRAIN_PATH,
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE), classes=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])



from keras.callbacks import ModelCheckpoint,EarlyStopping,History
history = History()
model_checkpoint = ModelCheckpoint('cnn111.hdf5', monitor='loss', save_best_only=True)
EarlyStopping=EarlyStopping(monitor='acc', patience=50, verbose=2, mode='auto')
callbacks = [
                history,
                model_checkpoint,
                EarlyStopping
            ]
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(directory=TEST_PATH,
                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE), classes=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])

# 运行模型
model.fit_generator(train_generator, epochs=2, validation_data=test_generator)

from keras.utils import np_utils
from sklearn import metrics
import numpy as np
import time
t0=time.time()
test = test_datagen.flow_from_directory(TEST_PATH,target_size=(IMAGE_SIZE, IMAGE_SIZE), classes=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
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

#import matplotlib.pyplot as plt
#
#
#plt.figure(12)
#plt.subplot(121)
#train_acc = history.history['acc']
#val_acc = history.history['val_acc']
#epochs = range(len(train_acc))
#plt.plot(epochs, train_acc, 'b',label='train_acc')
#plt.plot(epochs, val_acc, 'r',label='test_acc')
#plt.title('Train and Test accuracy')
#plt.legend()
#     
#plt.subplot(122)
#train_loss = history.history['loss']
#val_loss = history.history['val_loss']
#epochs = range(len(train_loss))
#plt.plot(epochs, train_loss, 'b',label='train_loss')
#plt.plot(epochs, val_loss, 'r',label='test_loss')
#plt.title('Train and Test loss')
#plt.legend()
#plt.show()




# 找一张图片进行预测验证
img = load_img(path=PREDICT_IMG, target_size=(IMAGE_SIZE, IMAGE_SIZE))
# 转换成numpy数组
x = img_to_array(img)
# 转换后的数组为3维数组(224,224,3),
# 而训练的数组为4维(图片数量, 224,224, 3),所以我们可扩充下维度
x = K.expand_dims(x, axis=0)
# 需要被预处理下
x = preprocess_input(x)
# 数据预测
result = model.predict(x, steps=1)
# 最后的结果是一个含有5个数的一维数组，我们取最大值所在的索引号，即对应'bus', 'dinosaur', 'flower', 'horse', 'elephant'的顺序
print("result:", K.eval(K.argmax(result)))














































## -*- coding: utf-8 -*-
#"""
#Created on Wed Aug  5 10:53:59 2020
#
#@author: Administrator
#"""
##导入库文件
#import pandas as pd
#import numpy as np
#import os
#import keras
#import matplotlib.pyplot as plt
#from keras.layers import Dense,GlobalAveragePooling2D
#from keras.applications import MobileNet
#from keras.preprocessing import image
#from keras.applications.mobilenet import preprocess_input
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Model
#from keras.optimizers import Adam
#
##导入模型
#base_model=MobileNet(weights='imagenet',include_top=False)
#x=base_model.output
#x=GlobalAveragePooling2D()(x)
#x=Dense(1024,activation='relu')(x)
#x=Dense(1024,activation='relu')(x)
#x=Dense(512,activation='relu')(x)
#preds=Dense(120,activation='softmax')(x)
#model=Model(inputs=base_model.input,outputs=preds)
#
#for i,layer in enumerate(model.layers):
#    print(i,layer.name)
#    
#for layer in model.layers:
#   layer.trainable=False
## or if we want to set the first 20 layers of the network to be non-trainable
#for layer in model.layers[:20]:
#   layer.trainable=False
#for layer in model.layers[20:]:
#   layer.trainable=True
#   
#   
#train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
#
#train_generator=train_datagen.flow_from_directory('path-to-the-main-data-folder',
#                                                target_size=(224,224),
#                                                color_mode='rgb',
#                                                batch_size=32,
#                                                class_mode='categorical',
#                                                shuffle=True)
#
#model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
## Adam optimizer
## loss function will be categorical cross entropy
## evaluation metric will be accuracy
#
#step_size_train=train_generator.n//train_generator.batch_size
#model.fit_generator(generator=train_generator,
#                  steps_per_epoch=step_size_train,
#                  epochs=10)
#    