import streamlit as st
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import time

st.set_page_config(page_title="Fake Logo Detection", layout="centered")

# ------------------- CSS -------------------
st.markdown("""
<style>

.title {
    text-align:center;
    font-size:38px;
    font-weight:bold;
    color:#38bdf8;
    margin-bottom:20px;
    animation: fadeIn 2s ease-in-out;
}

.center-container {
    display: flex;
    justify-content: center;
    align-items: center;
}

.center-container img {
    max-width: 250px;
    border-radius: 10px;
    display:block;
    margin:auto;
}

.result-card {
    background:#22c55e;
    color:white;
    padding:10px;
    border-radius:10px;
    font-size:20px;
    text-align:center;
    width:300px;
    margin:auto;
    margin-top:20px;
    animation: fadeIn 1s ease-in-out;
}

.confidence-card {
    background:#3b82f6;
    color:white;
    padding:8px;
    border-radius:10px;
    text-align:center;
    font-size:18px;
    width:300px;
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

# ------------------- HTML -------------------
st.markdown('<div class="title">Fake Logo Detection</div>', unsafe_allow_html=True)

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
    
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(image, width=250)
    st.markdown("</div>", unsafe_allow_html=True)

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

