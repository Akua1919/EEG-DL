#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Hide the Configuration and Warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import the Used Packages: Numpy, Pandas, and Tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

def eeg_cnn(input_shape):
    model=keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32,kernel_size = 3,strides = (1,1),padding = 'same',activation = tf.nn.relu,input_shape = input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2), padding = 'same'))
    model.add(keras.layers.Conv2D(filters=64,kernel_size = 3,strides = (1,1),padding = 'same',activation = tf.nn.relu))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2), padding = 'same'))
    #model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=512,activation = tf.nn.relu))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=3,activation = tf.nn.softmax))
    return model

def trian_model(train_data, train_labels, test_data, test_labels):
    train_data = np.reshape(train_data, [-1,60,1000,1])
    test_data = np.reshape(test_data, [-1,60,1000,1])

    model=eeg_cnn(input_shape=(60,1000,1))
    model.compile(optimizer='adam',loss="mse",metrics=['accuracy'])
    #model.summary()
    model.fit(x=train_data,y=train_labels,epochs=100,validation_data=(test_data,test_labels))


    # test_loss,test_acc=model.evaluate(x=train_data,y=train_labels)
    # print("Train Accuracy %.2f"%test_acc)
    test_loss,test_acc=model.evaluate(x=test_data,y=test_labels)
    print("Test Accuracy %.2f"%test_acc)
 
    cnt=0
    predictions=model.predict(test_data)
    for i in range(len(test_data)):
        target=np.argmax(predictions[i])
        label=np.argmax(test_labels[i])
        if target==label:
            cnt +=1
    print("correct prediction of total : %.2f"%(cnt/len(test_data)))
 
    model.save('./model/mnist-model.ckpt')


DIR = 'Saved_Matlab_Data/'

# Read Training Data
train_data = pd.read_csv(DIR + 'training_set_15.csv', header=None)
train_data = np.array(train_data).astype('float32')

# Read Training Labels
train_labels = pd.read_csv(DIR + 'training_label_15.csv', header=None)
train_labels = np.array(train_labels)

# Read Testing Data
test_data = pd.read_csv(DIR + 'valid_set_15.csv', header=None)
test_data = np.array(test_data).astype('float32')

# Read Testing Labels
test_labels = pd.read_csv(DIR + 'valid_label_15.csv', header=None)
test_labels = np.array(test_labels)

trian_model(train_data, train_labels, test_data, test_labels)