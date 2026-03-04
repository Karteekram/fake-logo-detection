import streamlit as st
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import time
import base64
import io
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

st.markdown(
    '<div class="title">An Enhanced Fake Logo Verification System using Vision Transformer</div>',
    unsafe_allow_html=True
)

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
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# ------------------- IMAGE CENTER -------------------
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

uploaded_file = st.file_uploader("Upload Logo Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    display_centered_image(image)

    image_tensor = transform(image).unsqueeze(0).to(device)

    loading_text = st.empty()

    loading_text.markdown(
        '<div class="loader-text">Analyzing Logo...</div>',
        unsafe_allow_html=True
    )

    time.sleep(2)

    with torch.no_grad():

        outputs = model(pixel_values=image_tensor)

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()

        # -------- EMBEDDING SIMILARITY --------
        features = outputs.hidden_states[-1][:,0,:].cpu().numpy()

        # dummy real reference vector (can be replaced with real dataset embeddings)
        reference = np.random.rand(1,768)

        similarity = cosine_similarity(features, reference)[0][0]

    loading_text.empty()

    # Combined decision
    if fake_prob > real_prob or similarity < 0.3:
        prediction = "Fake"
        confidence = max(fake_prob, 1-similarity)
    else:
        prediction = "Real"
        confidence = real_prob

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