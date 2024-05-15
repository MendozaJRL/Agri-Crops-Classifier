# Contents of AgriDeployment.py

import streamlit as st
import tensorflow as tf

@st.cache_resource
def getmymodel():
  model = tf.keras.models.load_model('Model83.h5')
  return model

model = getmymodel()
st.write("""# Agricultural Crops Classifier""")
file = st.file_uploader("Choose crop photo from computer",type = ["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np

def import_and_predict(image_data,model):
    size=(200,200)
    image=ImageOps.fit(image_data, size)
    img=np.asarray(image)
    img = img.astype('float32')
    img = img / 255.0
  
    #img_reshape=img[np.newaxis,...]
    prediction = model.predict(img)
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
