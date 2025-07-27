import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown

# Create working directory
working_dir = "model_files"
os.makedirs(working_dir, exist_ok=True)

model_filename = "Plant_diseases_Prediction_model.h5"
model_path = os.path.join(working_dir, model_filename)

# Google Drive file ID (replace with your actual ID)
file_id = "1On-_95vpaFT7l2TqirTNDe2SUHYugz1q"
url = f"https://drive.google.com/uc?id={file_id}"

# Download if not already present
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Updated path: class_indices.json is in root, not model_files
class_indices_path = "class_indices.json"
if not os.path.exists(class_indices_path):
    st.error("class_indices.json file is missing in the root directory.")
    st.stop()

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Image preprocessing
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Predict class
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit UI
st.title('🌿 Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((150, 150)), caption="Uploaded Image")

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'🧠 Prediction: **{prediction}**')
