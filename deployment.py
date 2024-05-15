# Contents of AgriDeployment.py

import streamlit as st
import tensorflow as tf

@st.cache_resource
# Load the Model
def load_model():
  model = tf.keras.models.load_model('Model83.h5')
  return model
myModel = load_model()

st.write("""# Agricultural Crops Classifier""")
file = st.file_uploader("Choose crop photo from computer",type = ["jpg","png"])

from numpy import argmax
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from PIL import Image,ImageOps

# Array that corresponds to the labels
Classification = ['Jute', 'Maize', 'Rice', 'Sugarcane', 'Wheat']
width = 64
length = 64

# load and prepare the image
def Prediction(filepath):
    # Load and prepare the image
    img = filepath

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
    test_image_path = Image.open(file)
    st.image(test_image_path,use_column_width=True)
    prediction = Prediction(test_image_path)

    string="OUTPUT : " + prediction
    st.success(string)
