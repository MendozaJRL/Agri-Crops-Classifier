# Contents of AgriDeployment.py

import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('Model83.h5')
  return model

model = load_model()
st.write("""# Agricultural Crops Classifier""")
file = st.file_uploader("Choose crop photo from computer",type = ["jpg","png"])

import cv2
import numpy as np
from numpy import argmax
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

Classification = ['Jute (Saluyot)', 'Maize (Mais)', 'Rice (Bigas)', 'Sugarcane (Tubo)', 'Wheat (Trigo)']

# load and prepare the image
def Prediction(filepath):
    # Load and prepare the image
    img = load_img(filepath, grayscale=True, target_size=(width, length))

    # Convert to array
    img = img_to_array(img)

    # Reshape into a single sample with 1 channel
    img = img.reshape(1, width, length, 1)

    # Prepare pixel data
    img = img.astype('float32')
    img = img / 255.0

    # Obtains a value or digit which will be the index for classification
    predict_value = myModel.predict(img)
    digit = argmax(predict_value)

    # Identify using the obtained digit
    return Classification[digit]

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    prediction = Prediction(image)

    string="OUTPUT : " + prediction
    st.success(string)
