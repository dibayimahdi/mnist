
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 19:15:10 2021
@author: Mahdi
"""

from keras.datasets import mnist
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense

from keras.losses import categorical_crossentropy
#from keras.losses import 
from keras.optimizers import Adam
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

X_train=train_images.reshape(60000,28,28,1)
X_test=test_images.reshape(10000,28,28,1)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255
X_test/=255



Y_train=np_utils.to_categorical(train_labels)
Y_test=np_utils.to_categorical(test_labels)

#model =  keras.Model(inputs=X_train, outputs=Y_train)
myinput=Input(shape=(28,28,1))
conv1=Conv2D(16,3,activation='relu',padding='same')(myinput)
Pool1=MaxPooling2D(pool_size=2)(conv1)
conv2=Conv2D(32,3,activation='relu',padding='same')(Pool1)
Pool2=MaxPooling2D(pool_size=2)(conv2)
flat=Flatten()(Pool2)
out_layer=Dense(10, activation='softmax')(flat)
mymodel= Model(myinput, out_layer)

mymodel.summary() 

mymodel.compile(optimizer=Adam(lr=.001),loss='categorical_crossentropy', metrics=['accuracy'])

network_history=mymodel.fit(X_train, Y_train,batch_size=128,epochs=3,validation_split=0.2)

history=network_history.history
acc=history['acc']
loss=history['loss']
val_acc=history['val_acc']
val_loss=history['val_loss']


plt.xlabel('epochs')
plt.ylabel('acc')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])

plt.figure()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)

plt.legend(['loss','val_loss'])

mymodel.evaluate(X_test,Y_test)
test_label_p=mymodel.predict(X_test)
test_label_p = np.argmax(test_label_p, axis=1)

