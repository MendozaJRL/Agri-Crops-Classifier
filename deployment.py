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
from PIL import Image,ImageOps
import numpy as np

def import_and_predict(image_data,model):
    width = 100
    height = 100
    size = (width, height)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    prediction = import_and_predict(image,model)

    class_names = ['Jute', 'Maize', 'Rice', 'Sugarcane', 'Wheat']

    string="OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
