import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

from chat_agent import ChatAgent
from reporting_agent import generate_medical_report

# ==========================
# Config
# ==========================
st.set_page_config(page_title="Lungs Vision Agent", layout="centered")
st.title("🧬 AI Lung Disease Vision Agent")

device = torch.device("cpu")

# ==========================
# Class Labels
# ==========================
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# ==========================
# Confidence Threshold
# ==========================
CONFIDENCE_THRESHOLD = 70.0  # below this → show warning, don't trust result

# ==========================
# Load Model (Inference Only)
# ==========================
@st.cache_resource
def load_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names)
    )
    model.load_state_dict(torch.load("best_covid_efficientnet.pth", map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()

# ==========================
# Transform (Match Training)
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Input Validation — Reject Non X-Ray Images
# ==========================
def is_valid_xray(image: Image.Image) -> bool:
    img_array = np.array(image)
    r = img_array[:, :, 0].astype(int)
    g = img_array[:, :, 1].astype(int)
    b = img_array[:, :, 2].astype(int)
    rg_diff = np.mean(np.abs(r - g))
    gb_diff = np.mean(np.abs(g - b))
    # X-rays are near-grayscale — channels should be very similar
    return rg_diff <= 20 and gb_diff <= 20

# ==========================
# Agents
# ==========================
chat_agent = ChatAgent()

# ==========================
# Upload Image
# ==========================
uploaded_image = st.file_uploader(
    "Upload a Lung X-ray image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_image:

    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # ── Input Validation ──────────────────────────
    if not is_valid_xray(image):
        st.error("❌ This does not appear to be a chest X-ray. "
                 "Please upload a valid grayscale chest X-ray image.")
        st.stop()

    img_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_prob, predicted_class = torch.max(probabilities, 1)

        condition             = class_names[predicted_class.item()]
        confidence_prob_value = confidence_prob.item()
        confidence_percent    = round(confidence_prob_value * 100, 2)

        result = {
            "predicted_condition": condition,
            "confidence_score":    confidence_prob_value,
            "confidence_percentage": confidence_percent
        }

    # ── Detection Result ──────────────────────────
    st.subheader("🩺 Detection Result")

    # Confidence threshold check
    if confidence_percent < CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠ Model confidence is low ({confidence_percent}%). "
            f"The prediction '{condition}' may not be reliable. "
            f"Please upload a clearer chest X-ray."
        )
    else:
        st.write(f"**Detected Condition:** {condition}")
        st.write(f"**Confidence Score:** {confidence_percent}%")

    st.info("⚠ This is AI-assisted analysis. Always consult a Doctor.")

    st.divider()

    # ── Generate Structured Medical Report ────────
    if st.button("📄 Generate Full Radiology Report"):
        with st.spinner("Generating structured medical report..."):
            report = generate_medical_report(result)

        st.subheader("📑 AI Radiology Report")
        st.text_area("Report Output", report, height=500)

        st.download_button(
            label="⬇ Download Report",
            data=report,
            file_name=f"report_{condition}.txt",
            mime="text/plain"
        )

    st.divider()

    # ── Chat Agent ────────────────────────────────
    st.subheader("💬 Ask Questions")

    user_question = st.text_input("Ask about the diagnosis, symptoms, or precautions")

    if user_question:
        with st.spinner("Thinking..."):
            answer = chat_agent.ask(user_question, result)
        st.write(answer)