# Contents of AgriDeployment.py

import streamlit as st
import tensorflow as tf

import cv2
from PIL import Image,ImageOps
import numpy as np

from numpy import argmax
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

@st.cache_resource
def getmymodel():
  model = tf.keras.models.load_model('Model83.h5')
  return model

model = getmymodel()
st.write("""# Agricultural Crops Classifier""")
file = st.file_uploader("Choose crop photo from computer",type = ["jpg","png"])

def import_and_predict(image_data, model):
    size=(200,200)
    image=ImageOps.fit(image_data, size)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
  
    img = img.astype('float32')
    img = img / 255.0
  
    predict_value = model.predict(img)
    digit = argmax(predict_value)
    return digit

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    prediction = import_and_predict(image, model)

    class_names = ['Jute (Saluyot)', 'Maize (Mais)', 'Rice (Bigas)', 'Sugarcane (Tubo)', 'Wheat (Trigo)']

    string="OUTPUT : " + class_names[prediction]
    st.success(string)
