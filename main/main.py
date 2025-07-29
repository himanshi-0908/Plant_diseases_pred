import os
import json
import urllib.request
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Setup paths
model_url = "https://huggingface.co/ranaHimanshi/plant-disease-model/resolve/main/plant_disease_prediction_model.h5"
model_filename = "plant_disease_prediction_model.h5"
model_path = os.path.join(".", model_filename)

# Download model if not present
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(model_url, model_path)

# Load the model
model = tf.keras.models.load_model(model_path)

# Load class indices (this must be uploaded to root of Hugging Face space)
class_indices_path = "class_indices.json"
if not os.path.exists(class_indices_path):
    st.error("❌ 'class_indices.json' is missing in the root directory.")
    st.stop()

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Preprocessing function
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_image_class(image):
    img_array = load_and_preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App UI
st.title("🌿 Plant Disease Prediction App")
st.markdown("Upload a clear image of a leaf to detect the disease.")

uploaded_file = st.file_uploader("📷 Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=200)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            prediction = predict_image_class(uploaded_file)
            st.success(f"🧠 Prediction: **{prediction}**")
