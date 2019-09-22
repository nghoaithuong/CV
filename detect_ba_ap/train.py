import tensorflow as tf
from tensorflow import keras
#import keras
from keras.preprocessing.image import ImageDataGenerator



import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns

import cv2
import os
import gc
import random

train_dir = '/home/hoaithuong/CV/detect_ba_ap/Train'

test_dir = '/home/hoaithuong/CV/detect_ba_ap/Test_Banana1'

train_bananas = ['/home/hoaithuong/CV/detect_ba_ap/Train/{}'.format(i) for i in os.listdir(train_dir) if 'banana' in i]
train_apples = ['/home/hoaithuong/CV/detect_ba_ap/Train/{}'.format(i) for i in os.listdir(train_dir) if 'apple' in i]
print(len(train_apples))
test_images = ['/home/hoaithuong/CV/detect_ba_ap/Test_Banana1/{}'.format(i) for i in os.listdir(test_dir)]

train_images = train_bananas + train_apples
random.shuffle(train_images)
print(len(train_images))
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
    y=[]  #labels
    for image in list_of_images:
        z=cv2.imread(image)
        z=cv2.cvtColor(z,cv2.COLOR_BGR2RGB)
        z=cv2.resize(z,(nrows, ncolumns))
        x.append(z)
        if 'banana' in image:
            y.append(1)
        elif 'apple' in image:
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
plt.title('labels for bananas and apples ')
print(x.shape)
print(y.shape)
ntrain=len(x)
batch_size=32
#setup the layers
model = keras.Sequential(
    [
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
                              steps_per_epoch=ntrain//32 ,
                              epochs=10)
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
#plt.show()

x_test, y_test = read_and_process_image(test_images[0:5])
X = np.array(x_test)
test_datagen = ImageDataGenerator(rescale=1./255)
i = 0
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(X, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('a banana ')
    else:
        text_labels.append('an apple')
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 5 == 0:
        break
plt.show()

#use cảmera to detect
