#深度学习007-Keras微调进一步提升性能  https://www.jianshu.com/p/5c766be9a9d7
import numpy as np
import os,random,shutil
np.random.seed(7)

# 1, 准备数据集
#1，指定一些超参数：
train_data_dir='../data/data2/train'  # 训练集目录/train'
val_data_dir='../data/data2/test' # keras中将测试集称为validation set
train_samples_num=5350 # train set中全部照片数
val_samples_num=2300
IMG_W,IMG_H,IMG_CH=50,50,3 # 单张图片的大小
batch_size=50 # 不能是32，因为2000/32不能整除，后面会有影响。
epochs=180  # 用比较少的epochs数目做演示，节约训练时间

#train_data_dir='./flowers/train'  # 训练集目录/train'
#val_data_dir='./flowers/test' # keras中将测试集称为validation set
#train_samples_num=1000 # train set中全部照片数
#val_samples_num=200
#IMG_W,IMG_H,IMG_CH=224,224,3 # 单张图片的大小
#batch_size=50 # 不能是32，因为2000/32不能整除，后面会有影响。
#epochs=5  # 用比较少的epochs数目做演示，节约训练时间


# 2，准备图片数据流
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( # 单张图片的处理方式，train时一般都会进行图片增强
        rescale=1. / 255, # 图片像素值为0-255，此处都乘以1/255，调整到0-1之间
        shear_range=0.2, # 斜切
        zoom_range=0.2, # 放大缩小范围
        horizontal_flip=True) # 水平翻转

train_generator = train_datagen.flow_from_directory(# 从文件夹中产生数据流
    train_data_dir, # 训练集图片的文件夹
    target_size=(IMG_W, IMG_H), # 调整后每张图片的大小
    batch_size=batch_size,
    class_mode='categorical') # 此处是二分类问题，故而mode是binary

# 3，同样的方式准备测试集
val_datagen = ImageDataGenerator(rescale=1. / 255) # 只需要和trainset同样的scale即可，不需增强
val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMG_W, IMG_H),
        batch_size=batch_size,
        class_mode='categorical')

# 4，构建模型
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers
from keras.models import Model


def build_model():
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_W, IMG_H, IMG_CH))
    # 此处我们只需要卷积层不需要全连接层，故而inclue_top=False,一定要设置input_shape，否则后面会报错
    # 这一步使用applications模块自带的VGG16函数直接加载了模型和参数，作为我们自己模型的“身子”

    # 下面定义我们自己的分类器，作为我们自己模型的“头”
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))  # 如果没有设置input_shape,这个地方报错，显示output_shape有很多None
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))  # 二分类问题

    top_model.load_weights('./smoke_model_weight')
    # 上面定义了模型结构，此处要把训练好的参数加载进来，

    my_model = Model(inputs=base_model.input, outputs=top_model(base_model.output))  # 将“身子”和“头”组装到一起
    # my_model就是我们组装好的完整的模型，也已经加载了各自的weights

    # 普通的模型需要对所有层的weights进行训练调整，但是此处我们只调整VGG16的后面几个卷积层，所以前面的卷积层要冻结起来
    for layer in my_model.layers[:17]:  # 25层之前都是不需训练的
        layer.trainable = False

    # 模型的配置 fine-Tune的核心是对原始的骨架网络（此处为VGG16）进行参数的微调，所以需要用非常小的学习率，而且要用SGD优化器。
    my_model.compile(loss='binary_crossentropy',
                     optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),  # 使用一个非常小的lr来微调
                     metrics=['acc'])
    return my_model



# 开始用train set来微调模型的参数
print('start to fine-tune my model')
my_model=build_model()
from keras.callbacks import ModelCheckpoint,EarlyStopping,History
history = History()
    
model_checkpoint = ModelCheckpoint('flow1.h5', monitor='loss', save_best_only=True)
EarlyStopping=EarlyStopping(monitor='acc', patience=50, verbose=2, mode='auto')
callbacks = [
                history,
                model_checkpoint,
                EarlyStopping
            ]

#model.fit_generator(
#            train,
#            steps_per_epoch=len(train.classes)/train.batch_size,
#            epochs=200,
#            callbacks=callbacks,
#            validation_data=test,
#            validation_steps=len(test.classes)/test.batch_size,
#            verbose=2)

history_ft = my_model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples_num // batch_size,  #2000/16=125批量
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=val_samples_num // batch_size)



## 3，同样的方式准备测试集
#val_datagen = ImageDataGenerator(rescale=1. / 255) # 只需要和trainset同样的scale即可，不需增强
#val_generator = val_datagen.flow_from_directory(
#        val_data_dir,
#        target_size=(IMG_W, IMG_H),
#        batch_size=batch_size,
#        class_mode='binary',
#        ,shuffle=False)
from keras.utils import np_utils
from sklearn import metrics
import numpy as np
#test = val_datagen.flow_from_directory('%s/test'%path,target_size=in_size,batch_size=batch_size,class_mode='categorical',shuffle=False)
y_pred_=my_model.predict_generator(val_generator, len(val_generator.classes)/val_generator.batch_size)
test_labels=np_utils.to_categorical(val_generator.classes)
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
# 画图，将训练时的acc和loss都绘制到图上
import matplotlib.pyplot as plt

def plot_training(history):
    plt.figure(12)

    plt.subplot(121)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, 'b', label='train_acc')
    plt.plot(epochs, val_acc, 'r', label='test_acc')
    plt.title('Train and Test accuracy')
    plt.legend()

    plt.subplot(122)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, 'b', label='train_loss')
    plt.plot(epochs, val_loss, 'r', label='test_loss')
    plt.title('Train and Test loss')
    plt.legend()

    plt.show()


plot_training(history_ft)

print(len(my_model.layers)) #查看模型的整个层数    20层
my_model.save('flowers.h5')