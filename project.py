import numpy as np
import pandas as pd
import tensorflow
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

Model=load_model(r"C&D.keras")

st.title("🐶Classification🐱")
upload=st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
button=st.button("PREDICT")
if button:
    if upload is not None:  
            img = Image.open(upload)
            image= img.resize((150,150))
            image_ex=np.expand_dims(image,axis=0)
            pred=Model.predict(image_ex)[0][0]
            if pred<=0.5:
                st.image(image=img)
                st.subheader("Meow Meow its a 🐱")
                ki=(r"C:\Users\KINGNICKS-DELL\Downloads\da\kitten-meowing-105618.mp3")
                st.audio(ki, format='audio/mp3', autoplay=True,start_time=7)
            elif pred>0.5:
                st.image(image=img)
                st.subheader("Bark Bark its a 🐶")
                do=(r"C:\Users\KINGNICKS-DELL\Downloads\da\dog-barking-70772.mp3")
                st.audio(do, format='audio/mp3', autoplay=True,start_time=15)
