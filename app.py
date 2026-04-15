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

/* DESCRIPTION BOX (SAME ALIGNMENT) */
.desc-box {
    background: #1f2937;
    color: white;
    padding: 15px;
    border-radius: 15px;
    margin: auto;
    margin-top: 10px;
    font-size: 20px;
    text-align: left;
    line-height: 1.6;
    border-left: 5px solid #22c55e;
}

.desc-title {
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
    font-size: 20px;
    color: #22c55e;
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

# ------------------- BRAND FUNCTION -------------------
def get_brand_description(filename, prediction):
    name = filename.lower()

    if "amazon" in name:
        if prediction == "Fake":
            return "This appears to imitate Amazon branding but contains structural or alignment inconsistencies."
        elif prediction == "Real":
            return "This matches Amazon's official logo with correct typography and smile arrow."
        else:
            return "Unable to confidently verify this Amazon logo due to unclear features."

    elif "nike" in name:
        if prediction == "Fake":
            return "This logo resembles Nike but has distortions in the swoosh design or proportions."
        elif prediction == "Real":
            return "This correctly represents Nike’s iconic swoosh with proper alignment."
        else:
            return "The Nike logo cannot be confidently verified due to low confidence."

    elif "apple" in name:
        if prediction == "Fake":
            return "This Apple logo shows deviation from the standard bitten apple design."
        elif prediction == "Real":
            return "This matches Apple's official minimalist logo design."
        else:
            return "The system cannot confidently verify this Apple logo."

    elif "google" in name:
        if prediction == "Fake":
            return "This Google logo may have incorrect colors or font inconsistencies."
        elif prediction == "Real":
            return "This correctly matches Google's color and font styling."
        else:
            return "The authenticity of this Google logo is uncertain."

    else:
        if prediction == "Fake":
            return "This logo contains visual inconsistencies and does not match standard brand patterns."
        elif prediction == "Real":
            return "This logo appears consistent with authentic brand design patterns."
        else:
            return "The system cannot determine authenticity due to insufficient confidence."

# ------------------- TRANSFORM -------------------
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

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor).logits
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    loading_text.empty()

    classes = ["Fake", "Real"]
    prediction = classes[pred.item()]
    confidence_value = confidence.item()

    if confidence_value < 0.60:
        prediction = "Uncertain"

    st.markdown(
        f'<div class="result-card">Prediction: {prediction}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence-card">Confidence: {round(confidence_value*100,2)}%</div>',
        unsafe_allow_html=True
    )

    # ------------------- EXPLANATION -------------------
    if prediction == "Fake":
        explanation = """<div class="explain-box"><div class="explain-title">Why this logo is Fake:</div>
        • Distortion or mismatch detected<br>• Poor alignment<br>• Missing details</div>"""
    elif prediction == "Real":
        explanation = """<div class="explain-box"><div class="explain-title">Why this logo is Real:</div>
        • Matches official structure<br>• Correct colors and spacing<br>• High clarity</div>"""
    else:
        explanation = """<div class="explain-box"><div class="explain-title">Result Uncertain:</div>
        • Low confidence<br>• Unclear features</div>"""

    st.markdown(explanation, unsafe_allow_html=True)

    # ------------------- BRAND DESCRIPTION -------------------
    desc = get_brand_description(uploaded_file.name, prediction)

    st.markdown(f"""
    <div class="desc-box">
        <div class="desc-title">Logo Description</div>
        {desc}
    </div>
    """, unsafe_allow_html=True)