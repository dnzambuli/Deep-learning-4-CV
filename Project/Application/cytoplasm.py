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
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# UI Title
st.title("Cell Segmentation from Microscope Images")

# Upload or Camera
st.sidebar.header("Upload microscope image")
upload_option = st.sidebar.radio("Choose input method:", ("Upload Image", "Use Camera"))

# Image input
image = None
thumbnail = None
if upload_option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload a cell image", type=["jpg", "png", "bmp", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # Show 64x64 thumbnail preview
        thumbnail = image.copy()
        thumbnail.thumbnail((64, 64))
        st.sidebar.image(thumbnail, caption="Preview", width=64)
else:
    camera_input = st.sidebar.camera_input("Take a photo")
    if camera_input:
        image = Image.open(camera_input).convert("RGB")

if image:
    # Display side-by-side view
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # ---- Preprocess and Predict ----
    def preprocess(img):
        img = img.convert("RGB")
        img = img.resize((256, 256), resample=Image.BILINEAR)
        img = np.array(img) / 255.0
        return np.expand_dims(img, axis=0)

    input_img = preprocess(image)
    prediction = model.predict(input_img)[0]  # shape: (256, 256, 1)

    # ---- Post-process Mask + Overlay ----
    if prediction.ndim == 3 and prediction.shape[-1] == 1:
        prediction = prediction[:, :, 0]

    mask = (prediction > 0.5).astype(np.uint8) * 255

    # Resize mask to original image size
    original_size = image.size
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Convert image to BGR
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Overlay red where mask is active
    overlay = image_bgr.copy()
    overlay[mask == 255] = [0, 0, 255]

    # Blend with original
    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)

    # Draw green contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 0), thickness=2)

    # Convert to PIL image
    final_overlay = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    final_pil = Image.fromarray(final_overlay)

    with col2:
        st.subheader("Segmented Cells")
        st.image(final_pil, use_container_width=True, caption="Segmentation Overlay")


st.markdown(
    """
    <hr style="margin-top: 50px;">
    <div style="text-align: center; color: gray;">
        <small>© Damunza3SmartMadMan 2025</small>
    </div>
    """,
    unsafe_allow_html=True
)