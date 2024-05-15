# Contents of AgriDeployment.py

import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
  model = tf.keras.models.load_model('filename1.h5')
  return model

model = load_model()
st.write("""# Agricultural Crops Classifier""")
file = st.file_uploader("Choose crop photo from computer",type = ["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np

def import_and_predict(image_data,model):
    size=(100,100)
    image=ImageOps.fit(image_data, size)
    img=np.asarray(image)
  
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
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
