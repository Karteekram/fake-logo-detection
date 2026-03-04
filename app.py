import streamlit as st
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import time
import base64
import io

st.set_page_config(
    page_title="An Enhanced Fake Logo Verification System using Vision Transformer",
    layout="centered"
)

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
}

.loader-text {
    text-align:center;
    font-size:18px;
    font-weight:bold;
    margin-top:10px;
    animation: blink 1s infinite;
}

@keyframes blink {
    0% {opacity:0.2;}
    50% {opacity:1;}
    100% {opacity:0.2;}
}

</style>
""", unsafe_allow_html=True)

# ------------------- TITLE -------------------
st.markdown(
    '<div class="title">An Enhanced Fake Logo Verification System using Vision Transformer</div>',
    unsafe_allow_html=True
)

device = torch.device("cpu")

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained("fake_logo_model")
    model.to(device)
    model.eval()
    return model

model = load_model()

# ------------------- TRANSFORM -------------------
base_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Augmentations for TTA
tta_transforms = [
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ]),
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ]),
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ]),
]

# ------------------- IMAGE CENTER FUNCTION -------------------
def display_centered_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()
    
    st.markdown(f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{img_str}" 
                 style="width:300px; max-width:90%; border-radius:12px;">
        </div>
    """, unsafe_allow_html=True)

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("Upload Logo Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    display_centered_image(image)

    loading_text = st.empty()

    loading_text.markdown(
        '<div class="loader-text">Analyzing Logo...</div>',
        unsafe_allow_html=True
    )

    time.sleep(2)

    all_probs = []

    with torch.no_grad():

        # Base prediction
        tensor = base_transform(image).unsqueeze(0).to(device)
        outputs = model(pixel_values=tensor).logits
        probs = F.softmax(outputs, dim=1)
        all_probs.append(probs)

        # TTA predictions
        for aug in tta_transforms:
            tensor = aug(image).unsqueeze(0).to(device)
            outputs = model(pixel_values=tensor).logits
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)

    # Average probabilities
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)

    fake_prob = avg_probs[0][0].item()
    real_prob = avg_probs[0][1].item()

    if fake_prob > real_prob:
        prediction = "Fake"
        confidence = fake_prob
    else:
        prediction = "Real"
        confidence = real_prob

    loading_text.empty()

    if confidence < 0.60:
        prediction = "Uncertain"

    st.markdown(
        f'<div class="result-card">Prediction: {prediction}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence-card">Confidence: {round(confidence*100,2)}%</div>',
        unsafe_allow_html=True
    )