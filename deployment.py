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

width = 100
length = 100
size = (width, length)

def import_and_predict(image_data,model):
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)

    img_reshape = img.reshape(1, width, length, 1)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    prediction = import_and_predict(image,model)

    class_names = ['Jute (Saluyot)', 'Maize (Mais)', 'Rice (Bigas)', 'Sugarcane (Tubo)', 'Wheat (Trigo)']

    string="OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
