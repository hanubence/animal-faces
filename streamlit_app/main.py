import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os

from utils import preprocess_image

animals = ['cat', 'dog', 'wild']

def load_model():
    loc = os.path.join(os.getcwd(), 'models', 'model.h5')
    localhost_load_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    model = tf.keras.models.load_model(loc, options=localhost_load_option)
    return model

model = load_model()

st.title('Image Classification App')

# State 1: Upload image or take one with the camera
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded or taken
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Classifying...")    
    
    # Preprocess the image to fit your model's input requirements
    processed_image = preprocess_image(image, target_size=(128, 128))  # Modify this line based on your model

    # Predict the class
    prediction = model.predict(processed_image)

    print(prediction)
    scores = tf.nn.softmax(prediction[0])  # Assuming a softmax final layer
    print(scores)

    # Get the highest probability class
    predicted_class = np.argmax(scores, axis=0)
    probability = np.max(scores)

    st.image(image, caption=f'{animals[predicted_class]} {probability*100:.2f}%', use_column_width=True)
