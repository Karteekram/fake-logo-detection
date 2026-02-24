import streamlit as st
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import time

st.set_page_config(page_title="An Enhanced Fake Logo Verification System using Vision Transformer", layout="centered")

# ------------------- CSS -------------------
st.markdown("""
<style>

.block-container {
    padding-top: 2rem !important;
}

.title {
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#38bdf8;
    margin-bottom:20px;
    animation: fadeIn 2s ease-in-out;
}

/* MOBILE + LAPTOP IMAGE CENTER FIX */
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}

[data-testid="stImage"] img {
    width: 300px !important;
    max-width: 90% !important;
    border-radius:12px;
    margin:auto !important;
}

.result-card {
    background:#22c55e;
    color:white;
    padding:15px;
    border-radius:15px;
    font-size:20px;
    text-align:center;
    margin:auto;
    
    margin-top:10px;
    animation: fadeIn 1s ease-in-out;
}

.confidence-card {
    background:#3b82f6;
    color:white;
    padding:15px;
    border-radius:15px;
    text-align:center;
    font-size:20px;
    margin:auto;
   
    margin-top:10px;
    animation: fadeIn 1.5s ease-in-out;
}

@keyframes fadeIn {
    from {opacity:0;}
    to {opacity:1;}
}

</style>
""", unsafe_allow_html=True)

# ------------------- TITLE -------------------
st.markdown('<div class="title">An Enhanced Fake Logo Verification System using Vision Transformer</div>', unsafe_allow_html=True)

device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained("fake_logo_model")
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload Logo Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # DISPLAY IMAGE NORMALLY (NO HTML WRAPPER)
    st.image(image)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing Logo..."):
        time.sleep(2)
        with torch.no_grad():
            outputs = model(pixel_values=image_tensor).logits
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)

    classes = ["Fake", "Real"]

    st.markdown(f'<div class="result-card">Prediction: {classes[pred.item()]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-card">Confidence: {round(confidence.item()*100,2)}%</div>', unsafe_allow_html=True)