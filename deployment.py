# Contents of AgriDeployment.py

# Import Necessary Files
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
  model = tf.keras.models.load_model('Model84.h5')
  return model

def preprocess_image(image_data):
  img = image_data.resize((100,100))
  img_array = np.array(img)
  if img_array.shape[-1] == 4:
    img_array = img_array[..., :3]
  img_array = img_array / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

def make_prediction(model, img_array):
  predictions = model.predict(img_array)
  return predictions

def display_prediction(predictions):
  class_names = ['Jute (Saluyot)', 'Maize (Mais)', 'Rice (Bigas)', 'Sugarcane (Tubo)', 'Wheat (Trigo)']
  predicted_class_digit = np.argmax(predictions[0])
  predicted_class = class_names[predicted_class_digit]
  return predicted_class

def main():
  model = load_model()

  st.title("Agricultural Crops Classifier")
  st.write("Please Upload a Crop Image")
  
  file = st.file_uploader("Select Image ", type=["jpg", "jpeg", "png"])

  if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    st.write(f"Shape of preprocessed image: {img_array.shape}")

    predictions = make_prediction(model, img_array)
    prediction_class = display_prediction(predictions)
    st.success("The predicted crop is: {predicted_class}")

if __name__ = "__main__"
  main()
  
