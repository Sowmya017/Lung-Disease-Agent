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
# Page Config
# ==========================
st.set_page_config(page_title="Hospital Dashboard", layout="centered")
st.title("🏥 AI Lung Disease Dashboard")

# ==========================
# Block Access Without Login
# ==========================
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()

# ==========================
# Logout Button
# ==========================
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Logout"):
        st.session_state.clear()
        st.switch_page("app.py")

# ==========================
# Patient Information Section
# ==========================
st.subheader("🧾 Patient Information")

name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
patient_id = st.text_input("Patient ID")

if st.button("Save Patient Details"):
    st.session_state.patient_info = {
        "name": name,
        "age": age,
        "gender": gender,
        "patient_id": patient_id
    }
    st.success("Patient details saved successfully.")

st.divider()

# ==========================
# Model Config
# ==========================
device = torch.device("cpu")

class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
CONFIDENCE_THRESHOLD = 70.0

# ==========================
# Load Model
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# X-ray Validation
# ==========================
def is_valid_xray(image: Image.Image) -> bool:
    img_array = np.array(image)
    r = img_array[:, :, 0].astype(int)
    g = img_array[:, :, 1].astype(int)
    b = img_array[:, :, 2].astype(int)
    rg_diff = np.mean(np.abs(r - g))
    gb_diff = np.mean(np.abs(g - b))
    return rg_diff <= 20 and gb_diff <= 20

# ==========================
# Upload Image
# ==========================
st.subheader("🩻 Upload Lung X-ray")

uploaded_image = st.file_uploader(
    "Upload a Lung X-ray image",
    type=["jpg", "png", "jpeg"]
)

chat_agent = ChatAgent()

if uploaded_image:

    if "patient_info" not in st.session_state:
        st.error("Please save patient details first.")
        st.stop()

    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if not is_valid_xray(image):
        st.error("❌ This does not appear to be a chest X-ray.")
        st.stop()

    img_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_prob, predicted_class = torch.max(probabilities, 1)

        condition = class_names[predicted_class.item()]
        confidence_prob_value = confidence_prob.item()
        confidence_percent = round(confidence_prob_value * 100, 2)

        result = {
            "predicted_condition": condition,
            "confidence_score": confidence_prob_value,
            "confidence_percentage": confidence_percent
        }

    st.subheader("🩺 Detection Result")

    if confidence_percent < CONFIDENCE_THRESHOLD:
        st.warning(
            f"Low confidence ({confidence_percent}%). "
            "Please upload a clearer X-ray."
        )
    else:
        st.success(f"Detected Condition: {condition}")
        st.write(f"Confidence: {confidence_percent}%")

    st.info("⚠ AI-assisted analysis. Consult a licensed doctor.")

    st.divider()

    # ==========================
    # Generate Report
    # ==========================
    if st.button("📄 Generate Full Radiology Report"):

        with st.spinner("Generating report..."):
            report = generate_medical_report(
                result,
                st.session_state.patient_info
            )

        st.subheader("📑 AI Radiology Report")
        st.text_area("Report Output", report, height=400)

        st.download_button(
            label="⬇ Download Report",
            data=report,
            file_name=f"{patient_id}_report.txt",
            mime="text/plain"
        )

    st.divider()

    # ==========================
    # Chat Agent
    # ==========================
# ==========================
# Chat Agent (Stateful)
# ==========================
st.subheader("💬 Medical Assistant Chat")

# Initialize memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Ask about diagnosis, symptoms, precautions...")

if st.button("Send"):

    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):

            answer = chat_agent.ask(
                user_question,
                result,
                st.session_state.chat_history
            )

        # Store in memory
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })

# Display Chat History
if st.session_state.chat_history:

    st.markdown("### 🗂 Conversation History")

    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**👩‍⚕️ You:** {chat['question']}")
        st.markdown(f"**🤖 Assistant:** {chat['answer']}")
        st.divider()