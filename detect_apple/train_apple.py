import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
from keras import Sequential

import numpy as np
import os
import gc
import matplotlib.pyplot as plt

train_dir = '/home/hoaithuong/CV/detect_apple/Train'
test_dir =  '/home/hoaithuong/CV/detect_apple/Test'

train_images = ['/home/hoaithuong/CV/detect_apple/Train/{}'.format(i) for i in os.listdir(train_dir)]
test_images = ['/home/hoaithuong/CV/detect_apple/Test/{}'.format(i) for i in os.listdir(test_dir)]

print(len(train_images))
print(len(test_images))

import matplotlib.image as mimg
for image in train_images[0:5]:
   img= mimg.imread(image)
   plt.imshow(img)
plt.show()

ncolumns = 28
nrows = 28
channels = 3

def process_images(list_images):
    x=[] #images
    y=[] #labels
    for image in list_images:
        z=cv2.imread(image)
        z=cv2.cvtColor(z,cv2.COLOR_BGR2RGB)
        z=cv2.resize(z,(nrows, ncolumns))
        x.append(z)
        if 'apple' in image:
            y.append(1)
        elif 'not apple' in image:
            y.append(0)
    return x,y
x,y = process_images(train_images)
# show 5 images
plt.figure(figsize=(10,10))
columns=5
for i in range(columns):
    plt.subplot(5/columns+1, columns, i+1)
    plt.suptitle('Resize Image')
    plt.grid(False)
    plt.imshow(x[i], cmap=plt.cm.binary)
plt.show()

del train_images
gc.collect()
