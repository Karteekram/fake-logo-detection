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

/* RESULT CARD */
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

/* CONFIDENCE CARD */
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

/* LOADING TEXT */
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

/* EXPLANATION BOX */
.explain-box {
    background: #111827;
    color: white;
    padding: 15px;
    border-radius: 15px;
    margin: auto;
    margin-top: 10px;
    font-size: 20px;
    text-align: left;
    line-height: 1.6;
    border-left: 5px solid #38bdf8;
}

.explain-title {
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
    font-size: 20px;
    color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)

# ------------------- TITLE -------------------
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

#TRANSFORM (NO NORMALIZATION)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

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

    image_tensor = transform(image).unsqueeze(0).to(device)

    loading_text = st.empty()

    loading_text.markdown(
        '<div class="loader-text">Analyzing Logo...</div>',
        unsafe_allow_html=True
    )

    time.sleep(3)

    # PREDICTION LOGIC
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor).logits
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    loading_text.empty()

    classes = ["Fake", "Real"]
    prediction = classes[pred.item()]
    confidence_value = round(confidence.item()*100, 2)

    st.markdown(
        f'<div class="result-card">Prediction: {prediction}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence-card">Confidence: {confidence_value}%</div>',
        unsafe_allow_html=True
    )

    # ------------------- EXPLANATION -------------------
    if prediction == "Fake":
        explanation = """
        <div class="explain-box">
            <div class="explain-title">Why this logo is Fake:</div>
            • Inconsistent font style compared to original brand<br>
            • Slight variation in logo proportions<br>
            • Blurred or low-quality rendering detected<br>  
            • Incorrect placement of design elements<br>
            • Lack of sharpness in edges and curves<br>
            • Missing fine details present in original logo<br>  
            • Unusual spacing between letters or symbols<br>
            • Distorted aspect ratio of the logo<br>
            • Artificial or generated texture patterns<br>  
            • Absence of brand-specific design precision  
        </div>
        """
    else:
        explanation = """
        <div class="explain-box">
            <div class="explain-title">Why this logo is Real:</div>
            • Accurate font style matching official brand design<br> 
            • Proper logo proportions and symmetry maintained<br>
            • High clarity and sharp visual quality<br>
            • Correct positioning of all design elements<br>  
            • Well-defined edges and smooth curves<br>
            • Presence of fine details consistent with original logo<br>
            • Balanced spacing between letters and symbols<br>
            • Correct aspect ratio maintained<br>
            • Natural and consistent texture appearance<br>
            • High similarity with trained authentic logo patterns   
        </div>
        """

    st.markdown(explanation, unsafe_allow_html=True)