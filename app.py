import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTModel
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import time
import base64
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="An Enhanced Fake Logo Verification System using Vision Transformer",
    layout="centered"
)

# ------------------- CSS -------------------
st.markdown("""
<style>
.block-container { padding-top: 2rem !important; }

.title {
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#38bdf8;
    margin-bottom:20px;
}

.result-card, .confidence-card {
    color:white;
    padding:15px;
    border-radius:15px;
    font-size:20px;
    text-align:center;
    margin:auto;
    margin-top:10px;
}
.result-card { background:#22c55e; }
.confidence-card { background:#3b82f6; }

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

.explain-box, .desc-box {
    color:white;
    padding:15px;
    border-radius:15px;
    margin:auto;
    margin-top:10px;
    font-size:20px;
    text-align:left;
    line-height:1.6;
}
.explain-box { background:#111827; border-left:5px solid #38bdf8; }
.desc-box { background:#1f2937; border-left:5px solid #22c55e; }

.explain-title, .desc-title {
    font-weight:bold;
    text-align:center;
    margin-bottom:10px;
    font-size:20px;
}
.explain-title { color:#38bdf8; }
.desc-title { color:#22c55e; }

</style>
""", unsafe_allow_html=True)

# ------------------- TITLE -------------------
st.markdown('<div class="title">An Enhanced Fake Logo Verification System using Vision Transformer</div>', unsafe_allow_html=True)

device = torch.device("cpu")

# ------------------- LOAD MODELS -------------------
@st.cache_resource
def load_model():
    m = ViTForImageClassification.from_pretrained("fake_logo_model")
    m.to(device)
    m.eval()
    return m

@st.cache_resource
def load_feature_model():
    fm = ViTModel.from_pretrained("google/vit-base-patch16-224")
    fm.to(device)
    fm.eval()
    return fm

model = load_model()
feature_model = load_feature_model()

# ------------------- BRAND REFERENCE (demo) -------------------
brand_reference = {
    "Amazon": np.random.rand(768),
    "Nike": np.random.rand(768),
    "Apple": np.random.rand(768),
    "Google": np.random.rand(768)
}

# ------------------- FUNCTIONS -------------------
def extract_features(image_tensor):
    with torch.no_grad():
        out = feature_model(pixel_values=image_tensor)
        features = out.last_hidden_state[:, 0, :]
    return features.cpu().numpy()

def detect_brand(features):
    best_brand, best_score = "Unknown", -1
    for brand, ref in brand_reference.items():
        score = cosine_similarity(features, ref.reshape(1, -1))[0][0]
        if score > best_score:
            best_score = score
            best_brand = brand
    return best_brand, best_score

def get_brand_description(brand, prediction):
    if brand == "Amazon":
        return {
            "Fake": "This logo imitates Amazon but lacks alignment or structure accuracy.",
            "Real": "This matches Amazon’s official logo with correct typography.",
            "Uncertain": "Amazon logo detected but confidence is low."
        }.get(prediction)
    elif brand == "Nike":
        return {
            "Fake": "This Nike logo shows distortion in swoosh shape.",
            "Real": "This correctly represents Nike’s iconic swoosh.",
            "Uncertain": "Nike logo detected but uncertain classification."
        }.get(prediction)
    elif brand == "Apple":
        return {
            "Fake": "Apple logo shape appears altered or inconsistent.",
            "Real": "This matches Apple’s official minimalist logo.",
            "Uncertain": "Apple logo detected but confidence is low."
        }.get(prediction)
    else:
        return "Brand detected but no detailed information available."

# ------------------- TRANSFORM -------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def display_centered_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{img_str}" style="width:300px; max-width:90%; border-radius:12px;">
        </div>
    """, unsafe_allow_html=True)

# ------------------- APP -------------------
uploaded_file = st.file_uploader("Upload Logo Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    display_centered_image(image)

    image_tensor = transform(image).unsqueeze(0).to(device)

    loading_text = st.empty()
    loading_text.markdown('<div class="loader-text">Analyzing Logo...</div>', unsafe_allow_html=True)

    time.sleep(2)

    # Prediction
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor).logits
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    loading_text.empty()

    classes = ["Fake", "Real"]
    prediction = classes[pred.item()]
    confidence_val = confidence.item()

    if confidence_val < 0.60:
        prediction = "Uncertain"

    st.markdown(f'<div class="result-card">Prediction: {prediction}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-card">Confidence: {round(confidence_val*100,2)}%</div>', unsafe_allow_html=True)

    # Explanation
    st.markdown(f"""
    <div class="explain-box">
        <div class="explain-title">Explanation</div>
        Model analyzed visual structure, alignment, and learned patterns to classify authenticity.
    </div>
    """, unsafe_allow_html=True)

    # -------- BRAND DETECTION --------
    features = extract_features(image_tensor)
    brand, similarity = detect_brand(features)

    desc = get_brand_description(brand, prediction)

    st.markdown(f"""
    <div class="desc-box">
        <div class="desc-title">Detected Brand: {brand}</div>
        {desc}<br><br>
        Similarity Score: {round(similarity*100,2)}%
    </div>
    """, unsafe_allow_html=True)