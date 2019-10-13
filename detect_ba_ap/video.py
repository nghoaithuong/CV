import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from keras.models import load_model

#Load the saved model
model = load_model('/home/hoaithuong/CV/detect_ba_ap/model_keras.h5')
video = cv2.VideoCapture(1)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
        im = im.resize((50,50))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        test_datagen = ImageDataGenerator(rescale=1./255)
        for batch in test_datagen.flow(img_array, batch_size=1):
       # prediction = int(model.predict(img_array)[0][0])
                prediction = model.predict(batch)
        #if prediction is > 0.5, which means a banana on the image, then show the frame in gray color. Else show the color frame.
        if prediction > 0.5:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                   break
video.release()
cv2.destroyAllWindows()
