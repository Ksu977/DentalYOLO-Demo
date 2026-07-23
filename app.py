import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

st.title("AI Dental Cavity Detection")

uploaded = st.file_uploader(
    "Upload dental X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded)

    results = model(image, conf=0.25)

    output = results[0].plot()

    st.image(output, caption="Detected cavities")