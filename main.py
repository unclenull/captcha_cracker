import numpy as np
import glob
import os
from math import ceil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import cv2
import string
import matplotlib.pyplot as plt
import pandas as pd
from utils import show_metrics


cwd = os.getcwd()
MODEL_PATH = os.path.join(cwd, "model.h5")
METRICS_PATH = os.path.join(cwd, "metrics.png")

samples = glob.glob('./dataset/dul/*.jpg')   # 获取所有样本图片
samples = samples[:500]
np.random.shuffle(samples)    # 将图片打乱

BATCH_SIZE = 32
nb_total = len(samples)
nb_train = int(nb_total * 0.9)
nb_valid = nb_total - nb_train

train_samples = samples[:nb_train]
test_samples = samples[nb_train:]


Classes = string.digits + string.ascii_lowercase + string.ascii_uppercase
ClassesCount = len(Classes)
LettersCount = 4

# CNN适合在高宽都是偶数的情况，否则需要在边缘补齐，把全体图片都resize成这个尺寸(高，宽，通道)
img_shape = (60, 160, 3)
input_image = Input(shape=img_shape)
 
#直接将验证码输入，做几个卷积层提取特征，然后把这些提出来的特征连接几个分类器（36分类，因为不区分大小写），
#输入图片
#用预训练的Xception提取特征,采用平均池化
base_model = Xception(input_tensor=input_image, weights='imagenet', include_top=False, pooling='avg')
 
#用全连接层把图片特征接上softmax然后36分类，dropout为0.5，因为是多分类问题，激活函数使用softmax。
#ReLU - 用于隐层神经元输出
#Sigmoid - 用于隐层神经元输出
#Softmax - 用于多分类神经网络输出
#Linear - 用于回归神经网络输出（或二分类问题）
predicts = [Dense(ClassesCount, activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(LettersCount)]
 
model = Model(inputs=input_image, outputs=predicts)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
 
class CustomCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        global Current_epoch
        Current_epoch = epoch
 
 
"""
Imggen = ImageDataGenerator(
    rotation_range=10,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.2,
)


gen = Imggen.flow_from_directory('dataset', save_to_dir='tmp', target_size=(img_shape[0], img_shape[1]))
gen.next()
exit()
"""

#misc.imread把图片转化成矩阵，
#misc.imresize重塑图片尺寸misc.imresize(misc.imread(img), img_size)  img_size是自己设定的尺寸
#ord()函数主要用来返回对应字符的ascii码，
#chr()主要用来表示ascii码对应的字符他的输入时数字，可以用十进制，也可以用十六进制。
 
def data_generator(images, batch_size): #样本生成器，节省内存
    # this first batch only returns shape
    yield np.zeros([batch_size] + list(img_shape)), [np.zeros(batch_size)] * LettersCount

    while True:
        # epoch begins
        np.random.shuffle(images)
        x, y = [], []
        for img in images:
            x.append(cv2.imread(img))  # 读取resize图片,再存进x列表
            real_num = img[-8:-4]
            y_list = []
            for i in real_num:
                y_list.append(Classes.index(i))

            y.append(y_list)
            if len(x) >= batch_size:
                try:
                    x = preprocess_input(np.array(x).astype(float))
                except Exception:
                    import pdb; pdb.set_trace()
                y = np.array(y)
                yield x, [y[:, i] for i in range(LettersCount)]
                x, y = [], []
        # epoch ends
        if len(x) > 0:  # the last batch doesn't have enough
            y = np.array(y)
            yield x, [y[:, i] for i in range(LettersCount)]


callbacks = [
    # CustomCallback(),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=5,
        verbose=1
    )
]

history = model.fit(
    data_generator(train_samples, BATCH_SIZE),
    steps_per_epoch=ceil(nb_train / BATCH_SIZE),
    epochs=99,
    validation_data=data_generator(test_samples, BATCH_SIZE),
    validation_steps=ceil(nb_valid / BATCH_SIZE),
    callbacks=callbacks,
)

model.load_weights(MODEL_PATH)
print('Training done.')
show_metrics(history, LettersCount, METRICS_PATH)
