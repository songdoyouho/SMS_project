# 匯入相關所需的模組
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import h5py
import glob
import time
from random import shuffle
from collections import Counter
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# 參數設定------------------------------------------------------------------------------------
map_characters = {0: 'yes', 1: 'no'}
img_width = 200 
img_height = 200
num_classes = len(map_characters) # 要辨識的種類
test_size = 0.1
imgsPath = "yes_or_no_small/train"
#--------------------------------------------------------------------------------------------
# 將訓練資料圖像從檔案系統中取出並進行
def load_pictures():
    pics = []
    labels = []
    
    for k, v in map_characters.items(): # k: 數字編碼 v: label
        # 把所有圖像檔的路徑捉出來
        pictures = [k for k in glob.glob(imgsPath + "/" + v + "/*")]       
        print(v + " : " + str(len(pictures))) # 看一下各有多少訓練圖像
        for i, pic in enumerate(pictures):
            tmp_img = cv2.imread(pic)
            # 由於OpenCv讀圖像時是以BGR (Blue-Green-Red), 我們把它轉置成RGB (Red-Green-Blue)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img = cv2.resize(tmp_img, (img_height, img_width)) # 進行resize            
            pics.append(tmp_img) # 塞到 array 裡
            labels.append(k)    
    return np.array(pics), np.array(labels)

# 取得訓練資料集與驗證資料集
def get_dataset(save=False, load=False):
    if load: 
        # 從檔案系統中載入之前處理保存的訓練資料集與驗證資料集
        h5f = h5py.File('dataset.h5','r')
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        h5f.close()
        
        # 從檔案系統中載入之前處理保存的訓練資料標籤與驗證資料集籤
        h5f = h5py.File('labels.h5', 'r')
        y_train = h5f['y_train'][:]
        y_test = h5f['y_test'][:]
        h5f.close()
    else:
        # 從最原始的圖像檔案開始處理
        X, y = load_pictures()
        y = keras.utils.to_categorical(y, num_classes) # 這裡只有分兩類
        
        # 將整個資料集切分為訓練資料集與驗證資料集 (90% vs. 10%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) 
        if save: # 保存尚未進行歸一化的圖像數據
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_test', data=X_test)
            h5f.close()
            
            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()
    
    # 進行圖像每個像素值的型別轉換與normalize成零到一
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test

# 取得訓練資料集與驗證資料集  
X_train, X_test, y_train, y_test = get_dataset(save=True, load=False)

# 建 model -------------------------------------------------------------------------------------
'''
from keras.applications import VGG16
# 使用 vgg16 model
VGG_16 = VGG16(weights='imagenet',
                  include_top=False, # 在這裡告訴 keras我們只需要卷積基底的權重模型資訊
                  input_shape=(200, 200, 3)) # 宣告我們要處理的圖像大小與顏色通道數

VGG_16.trainable = True # 解凍 "卷積基底"

# 所有層直到block4_pool都應該被凍結，而 block5_conv1，block5_conv2, block5_conv3 及 block5_pool則被解凍        
layers_frozen = ['block5_conv1','block5_conv2', 'block5_conv3', 'block5_pool']
for layer in VGG_16.layers:
    if layer.name in layers_frozen:
        layer.trainable = True
    else:
        layer.trainable = False
        
# 把每一層是否可以被"trainable"的flat print出來
for layer in VGG_16.layers:
    print("{}: {}".format(layer.name, layer.trainable))
'''
from keras.applications import ResNet50

# ResNet model
Res_Net = ResNet50(include_top=False,
                  weights='imagenet',
                  input_shape=(200, 200, 3)
                  )

Res_Net.summary()
'''
from keras.applications.inception_resnet_v2 import InceptionResNetV2
IRV2 = InceptionResNetV2(include_top=True,
                        weights='imagenet', 
                        input_tensor=None, 
                        input_shape=None, 
                        pooling=None, 
                        classes=2
                        )
'''
from keras import models
from keras import layers

model = models.Sequential() # 產生一個新的網絡模型結構
model.add(Res_Net)        # 把ResNet疊上去
model.add(layers.Flatten()) 
model.add(Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid')) 
#model.add(Dense(num_classes, activation='softmax'))

model.summary()

from keras import optimizers
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5), # 使用小的learn rate
              metrics=['acc'])
#-------------------------------------------------------------------------------------------------
# training model----------------------------------------------------------------------------------

history = model.fit(X_train, y_train,
         batch_size=30,
         epochs=100,
         validation_data=(X_test, y_test),
         shuffle=True,
         callbacks=[ModelCheckpoint('model.h5', save_best_only=True)], # 儲存最好的 model 來做 testing
         verbose=2
         )

def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics),'-o')                              
    plt.plot(history.history.get(val_metrics),'-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])

def show_train_and_val_result():
    # 透過趨勢圖來觀察訓練與驗證的走向 (特別去觀察是否有"過擬合(overfitting)"的現象)
    import matplotlib.pyplot as plt   
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plot_train_history(history, 'loss','val_loss')

    plt.subplot(1,2,2)
    plot_train_history(history, 'acc','val_acc')

    plt.show()

# testing-------------------------------------------------------------------------

import os
from pathlib import PurePath # 處理不同作業系統file path的解析問題 (*nix vs windows)

# 載入要驗證模型的數據
def load_test_set(path):
    pics, labels = [], []
    yes_and_no = os.listdir(path)
    all_pic_path = []
    for yes_or_no in yes_and_no:
        img_path = os.path.join(path, yes_or_no)
        img_pathes = os.listdir(img_path)
        if yes_or_no == 'yes':
            char_name = 0
        else:
            char_name = 1
        #if char_name in reverse_dict:
        for pic in img_pathes:
            pic = os.path.join(img_path, pic)
            temp = cv2.imread(pic)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp, (img_height,img_width)).astype('float32') / 255.
            pics.append(temp)
            labels.append(char_name)
            all_pic_path.append(pic)
    X_test = np.array(pics)
    y_test = np.array(labels)
    y_test = keras.utils.to_categorical(y_test, num_classes) # 進行one-hot編碼
    print("Test set", X_test.shape, y_test.shape)
    return X_test, y_test, all_pic_path

imgsPath = "yes_or_no_small/test"

#載入數據
X_valtest, y_valtest, all_pic_path = load_test_set(imgsPath)

# 預測與比對
from keras.models import load_model

# 把訓練時val_loss最小的模型載入
model = load_model('model.h5')

# 預測與比對
y_pred = model.predict_classes(X_valtest)
acc = np.sum(y_pred==np.argmax(y_valtest, axis=1))/np.size(y_pred)
test_index = np.argmax(y_valtest, axis=1)
print("Test accuracy = {}".format(acc))

# 預測結果視覺化
def show_wrong_images():
    for i, pic_path in enumerate(all_pic_path):
        #print(y_pred[i], test_index[i])
        if y_pred[i] != test_index[i]:
            image = cv2.imread(pic_path)
            #a = model.predict(pic.reshape(1, 200, 200, 3))[0]
            cv2.namedWindow("Image")   
            cv2.imshow("Image", image)   
            cv2.waitKey(0)  
            cv2.destroyAllWindows() 

show_wrong_images()
# --------------------------------------------------------------------------------------------------
# 圖像增強 data argmentation--------------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# 保存在訓練過程中比較好的模型
filepath="model-dtaug.h5"

# 保留"val_acc"最好的那個模型
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5), # 使用小的learn rate
              metrics=['acc'])

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=30),
                            steps_per_epoch=X_train.shape[0] // 10,
                            epochs=100,
                            validation_data=(X_test, y_test),
                            callbacks=[checkpoint],
                            verbose=2
                            )

# ---------------------------------------------------------------------------------------------------
# visualizing testing result-------------------------------------------------------------------------
from keras.models import load_model

# 把訓練時val_loss最小的模型載入
model = load_model('model-dtaug.h5')

y_pred = model.predict_classes(X_valtest)
acc = np.sum(y_pred==np.argmax(y_valtest, axis=1))/np.size(y_pred)
test_index = np.argmax(y_valtest, axis=1)
print("Test accuracy = {}".format(acc)) 

show_wrong_images()

# 每一種角色的正確率
import sklearn

# 使用sklearn的分類報告來看預測結果
y_pred = model.predict(X_valtest)

print('\n', sklearn.metrics.classification_report(np.where(y_valtest > 0)[1], 
                                                  np.argmax(y_pred, axis=1), 
                                                  target_names=list(map_characters.values())), sep='')

# show the confusion matrix
import seaborn as sns; sns.set()
import pandas as pd
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(np.where(y_valtest > 0)[1], np.argmax(y_pred, axis=1))
classes = list(map_characters.values())
df = pd.DataFrame(conf_mat, index=classes, columns=classes)

fig = plt.figure(figsize = (10,10))
sns.heatmap(df, annot=True, square=True, fmt='.0f', cmap="Blues")
plt.title('yes or no classification')
plt.xlabel('ground truth')
plt.ylabel('prediction')

plt.show()