import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model.h5")

class_names = ['Bird-Drop','Clean','Dusty','Electrical-Damage','Physical-Damage','Snow-Covered']

st.title("☀️ SolarGuard - Solar Panel Defect Detection")

uploaded = st.file_uploader("Upload Solar Panel Image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).resize((150,150))
    img_arr = np.array(img)/255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = model.predict(img_arr)
    result = class_names[np.argmax(pred)]

    st.image(img)
    st.success(f"Predicted Defect Type: {result}")