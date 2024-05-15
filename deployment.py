# Contents of AgriDeployment.py

# Import Necessary Files
import streamlit as st
import tensorflow as tf

import cv2
from PIL import Image,ImageOps
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

@st.cache_resource

# Functions
def load_model():
  model = tf.keras.models.load_model('filename1.h5')
  return model

def import_and_predict(image_data, model):
    img = image_data.resize((100,100))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
  
    prediction = model.predict(img_array)
    return prediction
  
# Main
model = load_model()

st.write("""# Agricultural Crops Classifier""")
file = st.file_uploader("Choose crop photo from computer",type = ["jpg", "jpeg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    prediction = import_and_predict(image, model)

    class_names = ['Jute (Saluyot)', 'Maize (Mais)', 'Rice (Bigas)', 'Sugarcane (Tubo)', 'Wheat (Trigo)']

    string="OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
