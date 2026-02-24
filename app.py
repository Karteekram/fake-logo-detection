import streamlit as st
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

st.set_page_config(page_title="Fake Logo Detection", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main-title {
    text-align: center;
    color: white;
    font-size: 50px;
    font-weight: bold;
    margin-top: 20px;
}

.upload-box {
    border: 2px dashed #38bdf8;
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    background-color: #1e293b;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #22c55e;
    color: white;
    font-size: 25px;
    text-align: center;
}

.confidence-box {
    padding: 15px;
    border-radius: 15px;
    background-color: #3b82f6;
    color: white;
    text-align: center;
    font-size: 20px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HTML TITLE ------------------
st.markdown('<p class="main-title">🕵️ Fake Logo Detection System</p>', unsafe_allow_html=True)

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

st.markdown('<div class="upload-box">Upload Logo Image Below</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor).logits
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    classes = ["Fake", "Real"]

    st.markdown(f'<div class="result-box">Prediction: {classes[pred.item()]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-box">Confidence: {round(confidence.item()*100,2)}%</div>', unsafe_allow_html=True)