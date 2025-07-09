import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Set Streamlit page settings
st.set_page_config(page_title="🐾 Animal Classifier", page_icon="🐾", layout="centered")

# Load model
model = load_model("animal_classification_model.h5")

# Class names
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
               'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
               'Panda', 'Tiger', 'Zebra']

# Title section
st.markdown("<h1 style='text-align: center; color: #6c5ce7;'>🐾 Animal Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of an animal, and I'll guess which one it is! 🧠</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload section
uploaded_file = st.file_uploader("📤 Upload an Animal Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized_img = image.resize((224, 224))

    # Show image and prediction side by side
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="📸 Your Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(resized_img) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = round(prediction[0][class_index] * 100, 2)

    with col2:
        st.markdown("### 🧠 Prediction")
        st.success(f"Predicted Animal: **{class_names[class_index]}**")
        st.write(f"Confidence: **{confidence}%**")
        st.balloons()
else:
    st.warning("📥 Upload an image above to classify the animal.")

# Clean footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>Made with ❤️ by Sanika Mane | AIML | 2025</p>", unsafe_allow_html=True)
