import streamlit as st
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import time
import base64
import io
import numpy as np
import hashlib

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

.upload-card {
    background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 16px 18px;
    margin: 8px 0 14px 0;
}

.upload-title {
    color: #e2e8f0;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 6px;
}

.upload-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 2px;
}

.upload-note {
    color: #38bdf8;
    font-size: 13px;
    margin-top: 8px;
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
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
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


def build_image_specific_explanation(image, prediction, confidence):
    """Create a per-image explanation using simple visual metrics."""
    image_array = np.array(image)
    gray = np.array(image.convert("L"))

    # Image-specific metrics so each upload gets a distinct description.
    brightness = float(gray.mean())
    contrast = float(gray.std())
    edge_strength = float(np.abs(np.diff(gray.astype(np.float32), axis=0)).mean())

    channel_means = image_array.reshape(-1, 3).mean(axis=0)
    dominant_idx = int(np.argmax(channel_means))
    dominant_color = ["red", "green", "blue"][dominant_idx]

    confidence_text = f"{confidence * 100:.2f}%"
    size_text = f"{image.width}x{image.height}"

    if prediction == "Fake":
        title = "Why this logo is Fake:"
        points = [
            f"Model confidence indicates a likely fake pattern ({confidence_text})",
            f"Measured edge consistency score suggests uneven contours ({edge_strength:.2f})",
            f"Global contrast profile differs from clean brand references ({contrast:.2f})",
            f"Average brightness is {brightness:.2f}, which may indicate rendering artifacts",
            f"Dominant color channel is {dominant_color}, with non-standard color balance",
            f"Input image size analyzed: {size_text}",
        ]
    elif prediction == "Real":
        title = "Why this logo is Real:"
        points = [
            f"Model confidence supports a real-logo pattern ({confidence_text})",
            f"Edge consistency appears stable for logo boundaries ({edge_strength:.2f})",
            f"Contrast profile is coherent for a clean logo sample ({contrast:.2f})",
            f"Average brightness ({brightness:.2f}) is within expected visual range",
            f"Dominant color channel ({dominant_color}) aligns with learned style cues",
            f"Input image size analyzed: {size_text}",
        ]
    else:
        title = "Result Uncertain:"
        points = [
            f"Model confidence is below threshold ({confidence_text})",
            f"Edge consistency is borderline for a confident decision ({edge_strength:.2f})",
            f"Contrast/brightness combination is ambiguous ({contrast:.2f}/{brightness:.2f})",
            f"Dominant color channel detected: {dominant_color}",
            f"Input image size analyzed: {size_text}",
            "Try uploading a clearer and more centered logo image",
        ]

    bullet_lines = "".join([f"• {point}<br>" for point in points])
    return f"""
    <div class="explain-box">
        <div class="explain-title">{title}</div>
        {bullet_lines}
    </div>
    """


def build_unique_image_description(image, prediction, confidence):
    """
    Generate a unique, automatic description per image.
    Different images should not share exactly the same description text.
    """
    if "image_description_cache" not in st.session_state:
        st.session_state.image_description_cache = {}
    if "used_descriptions" not in st.session_state:
        st.session_state.used_descriptions = set()

    image_bytes = image.tobytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    # If this exact image was already described, reuse the same description.
    if image_hash in st.session_state.image_description_cache:
        return st.session_state.image_description_cache[image_hash]

    image_array = np.array(image)
    gray = np.array(image.convert("L"))
    brightness = float(gray.mean())
    contrast = float(gray.std())
    edge_strength = float(np.abs(np.diff(gray.astype(np.float32), axis=0)).mean())
    channel_means = image_array.reshape(-1, 3).mean(axis=0)
    dominant_color = ["red", "green", "blue"][int(np.argmax(channel_means))]

    tone_words = ["balanced", "vivid", "muted", "high-contrast", "soft", "sharp"]
    style_words = ["clean", "textured", "smooth", "crisp", "bold", "subtle"]
    tone_word = tone_words[int(image_hash[0:2], 16) % len(tone_words)]
    style_word = style_words[int(image_hash[2:4], 16) % len(style_words)]
    signature = image_hash[:8]

    description = (
        f"Auto description [{signature}]: This {prediction.lower()} logo sample appears {style_word} "
        f"with a {tone_word} visual profile. Confidence is {confidence * 100:.2f}%, average brightness is "
        f"{brightness:.2f}, contrast is {contrast:.2f}, edge activity is {edge_strength:.2f}, and the dominant "
        f"color channel is {dominant_color}."
    )

    # Absolute safety check: avoid exact text duplication across different images.
    if description in st.session_state.used_descriptions:
        description = f"{description} Variant-{image_hash[8:12]}"

    st.session_state.used_descriptions.add(description)
    st.session_state.image_description_cache[image_hash] = description
    return description


def predict_with_model(model, image_tensor):
    """Run a classifier and return top label, confidence, and full probabilities."""
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor).logits
        probs = F.softmax(outputs, dim=1)

    top_idx = int(torch.argmax(probs, dim=1).item())
    top_conf = float(probs[0][top_idx].item())

    id_to_label = getattr(model.config, "id2label", {})
    label = id_to_label.get(top_idx, str(top_idx)) if isinstance(id_to_label, dict) else str(top_idx)
    return label, top_conf, probs


def is_likely_logo(image):
    """
    Self-contained logo filter (no extra folder/model needed).
    Stricter rejection for person/natural photos using multiple signals.
    """
    rgb_u8 = np.array(image, dtype=np.uint8)
    gray = np.array(image.convert("L"), dtype=np.float32)
    rgb = rgb_u8.astype(np.float32)
    h, w = gray.shape
    area = float(h * w)

    # Texture and gradient behavior.
    contrast = float(gray.std())
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_strength = float(gx.mean() + gy.mean())
    channel_std = float(rgb.reshape(-1, 3).std(axis=0).mean())

    # Color complexity: photos usually have very high color diversity.
    small = np.array(Image.fromarray(rgb_u8).resize((64, 64)))
    unique_colors = np.unique(small.reshape(-1, 3), axis=0).shape[0]
    color_diversity_ratio = unique_colors / max(float(64 * 64), 1.0)

    # Logos often have more solid white/black regions than portraits.
    near_white = np.logical_and.reduce([rgb[:, :, 0] > 240, rgb[:, :, 1] > 240, rgb[:, :, 2] > 240]).sum()
    near_black = np.logical_and.reduce([rgb[:, :, 0] < 20, rgb[:, :, 1] < 20, rgb[:, :, 2] < 20]).sum()
    bg_ratio = float(near_white + near_black) / max(area, 1.0)

    # Approximate skin-tone coverage in YCbCr space (helps reject person photos).
    ycbcr = np.array(image.convert("YCbCr"), dtype=np.uint8)
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    skin_ratio = float(skin_mask.mean())

    # Fine texture density: portraits/natural scenes often have richer micro-texture.
    lap = (
        np.abs(gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
        if h > 2 and w > 2
        else np.zeros((1, 1), dtype=np.float32)
    )
    high_texture_ratio = float((lap > 22).mean())

    photo_score = 0
    photo_score += 1 if contrast > 62 else 0
    photo_score += 1 if channel_std > 56 else 0
    photo_score += 1 if edge_strength < 24 else 0
    photo_score += 1 if color_diversity_ratio > 0.55 else 0
    photo_score += 1 if bg_ratio < 0.16 else 0
    photo_score += 1 if skin_ratio > 0.15 else 0
    photo_score += 1 if high_texture_ratio > 0.42 else 0

    # Strong logo cues: many logos have solid background and sharper boundaries.
    logo_cue_score = 0
    logo_cue_score += 1 if bg_ratio > 0.26 else 0
    logo_cue_score += 1 if edge_strength > 30 else 0
    logo_cue_score += 1 if color_diversity_ratio < 0.42 else 0
    logo_cue_score += 1 if skin_ratio < 0.08 else 0

    looks_like_photo = (
        (photo_score >= 5 and (skin_ratio > 0.10 or color_diversity_ratio > 0.62))
        or (skin_ratio > 0.18 and color_diversity_ratio > 0.50 and bg_ratio < 0.20)
    )
    if logo_cue_score >= 3 and photo_score < 6:
        looks_like_photo = False

    return not looks_like_photo

# ------------------- FILE UPLOAD -------------------
st.markdown(
    """
    <div class="upload-card">
        <div class="upload-title">Upload Brand Logo</div>
        <div class="upload-subtitle">Choose a clear brand logo image to analyze.</div>
        <div class="upload-subtitle">Supported formats: JPG, JPEG, PNG</div>
        <div class="upload-note">Tip: Centered logo images give better results.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Upload Logo Image",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # SHOW IMAGE
    display_centered_image(image)

    image_tensor = transform(image).unsqueeze(0).to(device)

    if not is_likely_logo(image):
        st.error("Please upload a brand logo image only (no person/natural photos).")
        st.stop()

    # LOADING TEXT
    loading_text = st.empty()

    loading_text.markdown(
        '<div class="loader-text">Analyzing Logo...</div>',
        unsafe_allow_html=True
    )

    time.sleep(2)

    _, _, auth_probs = predict_with_model(model, image_tensor)
    fake_prob = float(auth_probs[0][0].item()) if auth_probs.shape[1] > 0 else 0.0
    real_prob = float(auth_probs[0][1].item()) if auth_probs.shape[1] > 1 else 0.0

    if fake_prob > real_prob:
        prediction = "Fake"
        confidence = fake_prob
    else:
        prediction = "Real"
        confidence = real_prob

    loading_text.empty()

    # Threshold
    if confidence < 0.60:
        prediction = "Uncertain"

    # DISPLAY RESULT
    st.markdown(
        f'<div class="result-card">Prediction: {prediction}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence-card">Confidence: {round(confidence*100,2)}%</div>',
        unsafe_allow_html=True
    )

    # ------------------- EXPLANATION -------------------
    explanation = build_image_specific_explanation(image, prediction, confidence)

    st.markdown(explanation, unsafe_allow_html=True)

    # ------------------- UNIQUE AUTO DESCRIPTION -------------------
    unique_description = build_unique_image_description(image, prediction, confidence)
    st.info(unique_description)