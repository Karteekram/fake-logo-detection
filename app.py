import streamlit as st
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

st.title("Fake Logo Detection System")

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
    st.image(image)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor).logits
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    classes = ["fake", "real"]

    st.success(f"Prediction: {classes[pred.item()]}")
    st.info(f"Confidence: {round(confidence.item()*100,2)}%")
