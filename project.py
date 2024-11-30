
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow
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
            elif pred>0.5:
                st.image(image=img)
                st.subheader("Bark Bark its a 🐶")
            
            
