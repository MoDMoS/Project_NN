
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

#Loading up the Regression model we created
model_G = keras.models.load_model('Model_gender.h5')
model_A = keras.models.load_model('Model_Age.h5')

#Caching the model for faster loading
@st.cache


# Define the prediction function
def predict(uploaded_file):
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((48,48),Image.ANTIALIAS)
        img = np.float32(img)
        (R, G, B) = cv2.split(img)
        G = G.reshape(-1, 48, 48, 1)

        result_G = model_G.predict(G)
        result_A = model_A.predict(G)

        if(round(result_G[0][0])!=0) :
            result_G = "Female"
        else :
            result_G = "Male"
    return result_G, int(result_A[0][0])

st.title('Age & Gender perdiction')
st.header('Enter image for perdiction:')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.image(uploaded_file)


if st.button('Predict Price'):
    price = predict(uploaded_file)
    st.text("Gender : " + price[0] + '\nAge : ' + str(price[1]))