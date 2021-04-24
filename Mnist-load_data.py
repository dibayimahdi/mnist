# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 19:15:10 2021

@author: Mahdi
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import glob
import cv2
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

plt.imshow(train_images[5],cmap='binary')

path= 'D:/sample_dataset/test/'
images= glob.glob(path+"*.jpg")
x=[]
for img in images:
    image=cv2.imread(img)
    image=cv2.resize(image,(100,200))
    image=image/255
    x.append(image)

