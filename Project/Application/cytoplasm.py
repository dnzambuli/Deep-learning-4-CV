import streamlit as st
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import tensorflow as tf

# Load .env
load_dotenv()
MODEL_PATH = os.getenv("CYTOPLASM_LOCAL_MODEL_URL")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# UI Title
st.title("Cell Segmentation from Microscope Images")

# Upload or Camera
st.sidebar.header("Upload microscope image")
upload_option = st.sidebar.radio("Choose input method:", ("Upload Image", "Use Camera"))


# Handle image input
image = None
if upload_option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload a cell image", type=["jpg", "png", "bmp"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
else:
    camera_input = st.sidebar.camera_input("Take a photo")
    if camera_input:
        image = Image.open(camera_input).convert("RGB")

if image:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Preprocess for model
    def preprocess(img):
        img = img.convert("RGB")
        img = img.resize((256, 256), resample=Image.BILINEAR)
        img = np.array(img) / 255.0
        return np.expand_dims(img, axis=0)

    input_img = preprocess(image)

    # Predict segmentation mask
    prediction = model.predict(input_img)[0]

    # Process mask (assuming single-channel output)
    mask = (prediction > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, image.size)  # resize mask back to original image size
    mask_img = Image.fromarray(mask)

    # Display mask
    # 🔄 Side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Segmented Cells")
        st.image(mask_img, use_container_width=True, caption="Segmentation Output")
