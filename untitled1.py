import keras
from keras import Model
from keras.applications import VGG16
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import load_model
from keras.preprocessing import image
from PIL import ImageFile
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
 
EPOCHS = 30
BATCH_SIZE = 16
DATA_TRAIN_PATH = './flowers/train'
 
 
 
def Train():
    #-------------准备数据--------------------------
    #数据集目录应该是 train/LabelA/1.jpg  train/LabelB/1.jpg这样
    gen = ImageDataGenerator(rescale=1. / 255)  
    train_generator = gen.flow_from_directory(DATA_TRAIN_PATH, (224,224), shuffle=False,
                                batch_size=BATCH_SIZE, class_mode='categorical')

    #-------------加载VGG模型并且添加自己的层----------------------
    #这里自己添加的层需要不断调整超参数来提升结果，输出类别更改softmax层即可
 
    #参数说明：inlucde_top:是否包含最上方的Dense层，input_shape：输入的图像大小(width,height,channel)                                         
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x=Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    predictions = Dense(5, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
 
 
    #-----------控制需要FineTune的层数，不FineTune的就直接冻结
    for layer in base_model.layers:
        layer.trainable = False
 
    #----------编译，设置优化器，损失函数，性能指标
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()#显示模型
    #----------设置tensorboard,用来观察acc和loss的曲线---------------
    tbCallBack = TensorBoard(log_dir='./logs/' + TIMESTAMP,  # log 目录
                             histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             batch_size=16,  # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)
 
    #---------设置自动保存点，acc最好的时候就会自动保存一次，会覆盖之前的存档---------------
    checkpoint = ModelCheckpoint(filepath='HatNewModel.h5', monitor='acc', mode='auto', save_best_only='True')
 
    #----------开始训练---------------------------------------------
    model.fit_generator(generator=train_generator,
                        epochs=EPOCHS,
                        callbacks=[tbCallBack,checkpoint],
                        verbose=2
                        )
 
Train()
##-------------预测单个图像--------------------------------------
#def Predict(imgPath):
#    model = load_model(SAVE_MODEL_NAME)
#    img = image.load_img(imgPath, target_size=(224, 224))
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    res = model.predict(x)
#    print(np.argmax(res, axis=1)[0])