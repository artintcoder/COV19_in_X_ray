# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model=load_model(r'covid19.model')

face=cv2.imread ('img.jpg')


face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
face=cv2.resize(face,(224,224))
face=img_to_array(face)
face=preprocess_input(face)
face=np.expand_dims(face,axis=0)

(mask,withoutMask)=model.predict(face)[0]

#determine the class label and color we will use to draw the bounding box and text
print ('normal' ) if mask>withoutMask else print ('covid' )

