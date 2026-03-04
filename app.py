import streamlit as st
import cv2
import numpy as np
from PIL import Image

from utils.preprocess import preprocess_image
from model import load_model

st.title("Fake Currency Detection")

model = load_model()

img_file = st.camera_input("Show Currency Note")

if img_file is not None:

    img = Image.open(img_file)
    img = np.array(img)

    processed = preprocess_image(img)

    prediction = model.predict(processed)

    if prediction > 0.5:
        st.error("Fake Currency Detected")
    else:
        st.success("Real Currency Note")
