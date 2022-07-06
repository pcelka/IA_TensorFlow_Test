import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

import sys
import cv2

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

new_model = tf.keras.models.load_model('/home/philippe/Documents/Test_TF_on_new_model/my_model.h5')
class_names=['Bounty','Maltesers', 'Snickers','Twix']
#new_model.summary()
img_height = 250
img_width = 250

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        # Cropping an image Y1:Y2 , X1:X2
        cropped_image = frame[115:365, 140:390]
        cv2.imwrite("test.png", cropped_image)
        print("written!")
        img = tf.keras.utils.load_img('/home/philippe/Documents/Test_TF_on_new_model/test.png',target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = new_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))) 
        cv2.waitKey(1)
cam.release()
cv2.destroyAllWindows()




