# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 19:15:10 2021

@author: Mahdi
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import categorical_crossentropy
#from keras.losses import 
from keras.optimizers import SGD
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

X_train=train_images.reshape(60000,784)
X_test=test_images.reshape(10000,784)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255
X_test/=255



Y_train=np_utils.to_categorical(train_labels)
Y_test=np_utils.to_categorical(test_labels)

mymodel=Sequential()
mymodel.add(Dense(500,activation='relu',input_shape=(784,)))

mymodel.add(Dense(100,activation='relu'))

mymodel.add(Dense(10,activation='softmax'))

mymodel.summary()

mymodel.compile(optimizer=SGD(lr=.001),loss='categorical_crossentropy', metrics=['accuracy'])

network_history=mymodel.fit(X_train, Y_train,batch_size=128,epochs=3)

history=network_history.history
acc=history['acc']
loss=history['loss']

plt.plot(acc)
plt.plot(loss)

