# Contents of AgriDeployment.py

# Import Necessary Files
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
  model = tf.keras.models.load_model('Model79.4.h5')
  return model

def prepare_image(image_data):
  img = image_data.resize((100,100))
  img_array = np.array(img)
  if len(img_array.shape) > 2:
    img_array = np.mean(img_array, axis=2)
  img_array = img_array.reshape(1, 100, 100, 1)
  img_array = img_array.astype('float32')
  img_array = img_array / 255.0
  return img_array

def prediction(model, img_array):
  predictions = model.predict(img_array)
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
    img_array = prepare_image(image)
    
    result = prediction(model, img_array)
    st.success(f"The predicted crop is: {result}")

if __name__ == "__main__":
  main()
  
