import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Load the trained model
model = load_model('skin_cancer_cnn.h5')

# Custom CSS styling for medical dashboard feel
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/cubes.png');
        background-size: cover;
        background-attachment: fixed;
    }
    html, body, [cass*="css"] {
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
    }
    .title {
        font-size: 48px;
        color: #0a3d62;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .benign {
        color: green;
        font-size: 22px;
        font-weight: bold;
        padding: 10px 0;
    }
    .malignant {
        color: red;
        font-size: 22px;
        font-weight: bold;
        padding: 10px 0;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: gray;
        margin-top: 30px;
    }
    .confidence {
        font-size: 18px;
        color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Branding
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
st.sidebar.title("SkinCheck AI")
st.sidebar.markdown("An early detection support tool for skin cancer classification using deep learning.")

# Title
st.markdown('<div class="title">🩺 Skin Cancer Detection App</div>', unsafe_allow_html=True)

# Description
st.markdown("""
<div class="info-box">
    <h4>📄 Description</h4>
    <p>
        Upload a high-quality image of a skin lesion, and our AI model will classify it as either
        <b style="color: green;">Benign 🟢</b> or 
        <b style="color: red;">Malignant 🔴</b>. 
        This tool is designed to assist in early detection and risk assessment of skin cancer.
    </p>
</div>
""", unsafe_allow_html=True)

# Upload Image
uploaded_image = st.file_uploader("📤 Upload a skin lesion image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# Prediction Function
def predict_skin_cancer(image_file, model):
    img = Image.open(image_file).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "Malignant" if prediction > 0.5 else "Benign"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Predict and display result
if uploaded_image is not None:
    st.image(uploaded_image, caption="🖼️ Uploaded Skin Lesion", use_container_width=True)

    with st.spinner('🔍 Analyzing the image...'):
        time.sleep(1.5)
        label, confidence = predict_skin_cancer(uploaded_image, model)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("🧪 Prediction Result")

    if label == "Benign":
        st.markdown('<p class="benign">✅ Result: Benign (Non-cancerous)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="malignant">⚠️ Result: Malignant (Potentially cancerous)</p>', unsafe_allow_html=True)

    st.markdown(f'<p class="confidence">📈 Confidence: {confidence * 100:.2f}%</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Model Info
st.markdown("""
<div class="info-box">
    <h4>📊 Model Information</h4>
    <ul>
        <li><b>Type:</b> Convolutional Neural Network (CNN)</li>
        <li><b>Framework:</b> TensorFlow / Keras</li>
        <li><b>Trained On:</b> Skin lesion dataset (224x224 input)</li>
        <li><b>Output:</b> Benign 🟢 or Malignant 🔴 classification</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# How to Use
st.markdown("""
<div class="info-box">
    <h4>🧠 How to Use</h4>
    <ol>
        <li>Upload a clear, focused image of the skin lesion.</li>
        <li>Wait for the AI model to analyze the image.</li>
        <li>Review the result and confidence score.</li>
        <li>Consult a dermatologist for medical diagnosis.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Limitations
st.markdown("""
<div class="info-box">
    <h4>⚠️ Limitations & Disclaimer</h4>
    <p>
        This application is <strong>not a substitute for professional medical advice</strong>. 
        The model may not perform well on poor-quality, blurry, or highly variable lighting conditions.
        Always consult a certified dermatologist for diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)

# Learn More / Resources
st.markdown("""
<div class="info-box">
    <h4>📚 Learn More</h4>
    <ul>
        <li><a href="https://www.aad.org/public/diseases/skin-cancer/types/common" target="_blank">American Academy of Dermatology: Skin Cancer Types</a></li>
        <li><a href="https://www.who.int/news-room/fact-sheets/detail/skin-cancers" target="_blank">WHO Fact Sheet: Skin Cancers</a></li>
        <li><a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6504164/" target="_blank">CNNs in Dermatology - Research Paper</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 SkinCheck AI · Built with for early skin cancer awareness and support · Not a diagnostic tool.</div>', unsafe_allow_html=True)   