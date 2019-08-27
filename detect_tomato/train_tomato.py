import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img



import numpy as np
import matplotlib.pyplot as plt
import matplotlib


import seaborn as sns

import cv2
import glob
import os
import gc
import random
import pandas as pd

train_dir = '/home/hoaithuong/CV/detect_tomato/Train_Tomato'
test_dir = '/home/hoaithuong/CV/detect_tomato/Test_Tomato'

train_images = ['/home/hoaithuong/CV/detect_tomato/Train_Tomato/{}'.format(i) for i in os.listdir(train_dir)]
test_images = ['/home/hoaithuong/CV/detect_tomato/Test_Tomato/{}'.format(i) for i in os.listdir(test_dir)]

print(len(train_images))
name = ['tomato']
random.shuffle(train_images)
gc.collect()

import matplotlib.image as mpimg
for ima in train_images[0:5]:
    img= mpimg.imread(ima)
    plt.imshow(img)
    plt.show()
nrows = 50
ncolumns = 50
channels = 3
def read_and_process_image(list_of_images):
    x=[] #images
    y=[]  #labelsqqqq
    for image in list_of_images:
        z=cv2.imread(image)
        z=cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        z=cv2.resize(z,(nrows, ncolumns))
        x.append(z)
        if 'tomato' in image:
            y.append(1)
        elif ' not tomato' in image:
            y.append(0)
    return x,y
x,y = read_and_process_image(train_images)
#print(x[0]) 
print(y)
#print(x.shape)
#Show 5 image
plt.figure(figsize=(20,10))
columns=5
for i in range(columns):
    plt.subplot(5/columns+1, columns, i+1)
    plt.imshow(x[i])
    plt.show()

del train_images
gc.collect()
x=np.array(x)
y=np.array(y)
sns.countplot(y)
plt.title('labels for tomato ')
print(x.shape)
print(y.shape)
ntrain=len(x)
batch_size=32
#setup the layers
model = keras.Sequential(
    [
        #keras.layers.Flatten(input_shape=(50, 50, 3)),
        keras.layers.Conv2D(32,(3,3), activation='relu',input_shape=(50, 50, 3)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64,(3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(128,(3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(128,(3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ]
)
rms= keras.optimizers.RMSprop(lr=1e-4)
model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator( rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)
train_generator = train_datagen.flow(x, y, batch_size= batch_size)
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain,
                              epochs=5)
model.save_weights('model_weights.h5')
model.save('model_keras.h5')
acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

#Train accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.title('Training accurarcy')
plt.legend()

plt.figure()
#Train loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()

x_test, y_test = read_and_process_image(test_images[0:5])
X = np.array(x_test)
test_datagen = ImageDataGenerator(rescale=1./255)
i = 0
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('tomato')
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 5 == 0:
        break
plt.show()
