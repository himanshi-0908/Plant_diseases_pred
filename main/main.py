import os
import json
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# --- 1. Setup
working_dir = "model_files"
os.makedirs(working_dir, exist_ok=True)

model_filename = "plant_disease_prediction_model.h5"
model_path = os.path.join(working_dir, model_filename)

hf_model_url = "https://huggingface.co/ranaHimanshi/plant-disease-model/resolve/main/plant_disease_prediction_model.h5"

# --- 2. Download model from Hugging Face if not exists
if not os.path.exists(model_path):
    response = requests.get(hf_model_url, stream=True)
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        st.error("⚠️ Failed to download model from Hugging Face.")
        st.stop()

# --- 3. Load the model
try:
    from keras.models import load_model  # safer import for compatibility
    model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"🚫 Model loading failed: {e}")
    st.info("Try downgrading Keras using: `pip install keras==2.11.0`")
    st.stop()

# --- 4. Load class indices
class_indices_path = "class_indices.json"
if not os.path.exists(class_indices_path):
    st.error("❌ class_indices.json file is missing in the root directory.")
    st.stop()

with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# --- 5. Image preprocessing
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# --- 6. Prediction function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# --- 7. Streamlit UI
st.title('🌿 Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((150, 150)), caption="Uploaded Image")

    with col2:
        if st.button('Classify'):
            try:
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'🧠 Prediction: **{prediction}**')
            except Exception as e:
                st.error(f"Prediction failed: {e}")
