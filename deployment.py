# Contents of AgriDeployment.py

import streamlit as st
import tensorflow as tf

from numpy import argmax
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow.keras.models import model_from_json

@st.cache_resource
def Prediction(_filepath):   
    img = load_img(_filepath, color_mode="grayscale", target_size=(width, length))
  
    img = img_to_array(img)
    img = img.reshape(1, width, length, 1)
    img = img.astype('float32')
    img = img / 255.0

    myModel = model
    predict_value = myModel.predict(img)
    digit = argmax(predict_value)
    return class_names[digit]

# Main
st.write("""# Agricultural Crops Classifier""")
file = st.file_uploader("Choose crop photo from computer", type = ["jpg", "png"])

import cv2
from PIL import Image,ImageOps
import numpy as np

model = load_model("filename1.h5")
class_names = ['Jute (Saluyot)', 'Maize (Mais)', 'Rice (Bigas)', 'Sugarcane (Tubo)', 'Wheat (Trigo)']
width = 100
length = 100

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
  
    result = Prediction(image)

    string="OUTPUT : " + result
    st.success(string)
