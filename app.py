import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set title
st.title("Rice Leaf Disease Classification 🌾")
st.write("Upload a leaf image to detect its disease class.")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("rice_leaf_disease_model_v2.h5")
    return model

model = load_model()

# Define class names (must match order used during training)
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):  # <-- Ubah jadi 224x224
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  # normalize
    image_array = np.expand_dims(image_array, axis=0)  # add batch dim
    return image_array

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess & predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show result
    st.markdown("### Prediction Result")
    st.success(f"Predicted Class: **{predicted_class}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")
