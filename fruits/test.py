import tensorflow
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import load_model
import random
import matplotlib.pyplot as plt
import os
import h5py

model = load_model('/home/hoaithuong/PycharmProjects/fruits/model_keras.h5')
model.load_weights('/home/hoaithuong/PycharmProjects/fruits/model_weights.h5')

test_dir = '/home/hoaithuong/PycharmProjects/fruits/test'
test_images = ['/home/hoaithuong/PycharmProjects/fruits/test/{}'.format(i) for i in os.listdir(test_dir)]
random.shuffle(test_images)

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
        if 'apple' in image:
            y.append(0)
        if 'banana' in image:
            y.append(1)
        elif 'plum' in image:
            y.append(2)
    return x,y
x_test, y_test = read_and_process_image(test_images[0:9])
print(y_test)
#sys.exit()
X = np.array(x_test)
X= X/255.0
test_datagen = ImageDataGenerator(rescale=1./255)
#for batch in test_datagen.flow(X, batch_size=1):
prediction = model.predict(X)
print(prediction)
#pred=np.round(prediction)
#pred=pred.astype(int)
#pred = np.array(pred)
pred= np.array(prediction)
print(pred)
    #break
#sys.exit()
class_names = ['apple', 'banana', 'plum']
def plot_image(i, predictions_array, true_label, img):
    prediction_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predict_label = np.argmax(predictions_array)
    print(predict_label)
    print(true_label)
    print(predictions_array)
    if predict_label == true_label:
        color='blue'
    else:
        color='red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predict_label],
                                        100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color='blue')

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.grid(False)
    plt.xticks(range(3))
    plt.yticks([])
    thisplot=plt.bar(range(3), predictions_array, color="#777777")
    plt.ylim([0,1])
    predict_label = np.argmax(predictions_array)
    thisplot[predict_label].set_color('red')
    thisplot[true_label].set_color('blue')


#plot several images with their predictions
num_rows=3
num_cols=3
num_images=num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, pred[i], y_test, X)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, pred[i], y_test)
    plt.tight_layout()
plt.show()
