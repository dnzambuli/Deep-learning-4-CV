import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import gdown
import urllib.request
import tensorflow as tf
from tensorflow import keras 
import os
from dotenv import load_dotenv
import sys
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN if preferred

# Configure environment for PyInstaller compatibility
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    os.environ['GDOWN_SHOW_PROGRESS'] = "False"  # Disable tqdm progress bars
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages


load_dotenv()
MODEL_GDRIVE_URL = os.getenv("CYTOPLASM_MODEL_URL")

# Constants
IMG_SIZE = (256, 256)
MODEL_PATH = "pap_smear_model.keras"


# Load model
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model...")
            gdown.download(
                url=MODEL_GDRIVE_URL,
                output=MODEL_PATH,
                quiet=True,  # Disable progress bar
                fuzzy=True   # More flexible URL matching
            )
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Model download failed")
        
        print("Loading model...")
        model = keras.models.load_model(MODEL_PATH)
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Add fallback behavior here if needed
        return None


# Predict and show contours
def process_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, IMG_SIZE)
        img_input = img / 255.0
        img_tensor = np.expand_dims(img_input, axis=0)

        prediction = model.predict(img_tensor)
        predicted_mask = (prediction > 0.5).astype(np.uint8)[0, :, :, 0]

        contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_with_border = img.copy()
        cv2.drawContours(img_with_border, contours, -1, (0, 255, 0), thickness=2)

        # Show the result in a new window
        cv2.imshow("Predicted Cytoplasm Contours", img_with_border)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")

# Upload Image
def upload_image():
    filepath = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
    )
    if filepath and os.path.isfile(filepath):
        try:
            process_image(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

# Use Webcam
def capture_from_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows for more stable access
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        messagebox.showerror("Error", "Failed to capture image from camera.")
        return

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, frame)

        process_image(temp_path)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process captured image:\n{str(e)}")
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

# GUI
root = tk.Tk()
root.title("Cell Cytoplasm Detection")

tk.Label(root, text="Upload a microscope image or use camera").pack(pady=10)
tk.Button(root, text="Upload Image", command=upload_image).pack(pady=5)
tk.Button(root, text="Capture from Camera", command=capture_from_camera).pack(pady=5)

# Load model at startup
model = load_model()

root.mainloop()
