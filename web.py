import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# PAGE CONFIG
st.set_page_config(
    page_title="OIL PALM DISEASE CLASSIFICATION SYSTEM",
    page_icon="🌿",
    layout="centered"
)

# CSS THEME
st.markdown("""
<style>

.stApp {
    background: linear-gradient(165deg,#EEEE90, #C7F6C7, #064e3b);
    color: black;
}

/* TITLE */
h1 {
    text-align: center;
    color: #006b3c;
    font-weight: 800;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.4);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    border: 1px solid rgba(0,0,0,0.1);
}

/* BUTTON */
.stButton > button {
    width: 100%;
    background: #00b36b;
    color: black;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    padding: 10px;
}

/* IMAGE */
img {
    border-radius: 14px;
    border: 2px solid rgba(0,255,157,0.3);
}

/* METRICS */
[data-testid="stMetricValue"] {
    color: black;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.title("Oil Palm Disease Classification System")
st.markdown("---")

# SESSION STATE
if "image" not in st.session_state:
    st.session_state.image = None

if "result" not in st.session_state:
    st.session_state.result = None

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

# LOAD MODEL
@st.cache_resource
def load_model():
    checkpoint = torch.load("DenseNet121_best.pth", map_location="cpu"，weights_only=False)

    classes = checkpoint["label_classes"]

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes

model, classes = load_model()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# PREDICT FUNCTION
def predict(image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return classes[pred.item()], conf.item()

# RESET FUNCTION
def reset():
    st.session_state.image = None
    st.session_state.result = None
    st.session_state.upload_key += 1   

# UPLOAD SECTION
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("Upload Oil Palm Image")

upload_file = st.file_uploader(
    "Upload image for analysis",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.upload_key}"  # 🔥 IMPORTANT FIX
)

if upload_file is not None:
    st.session_state.image = Image.open(upload_file).convert("RGB")
    st.session_state.result = None

st.markdown("</div>", unsafe_allow_html=True)

# IMAGE + ACTION SECTION
if st.session_state.image is not None:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.image(st.session_state.image, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run Analysis"):
            label, conf = predict(st.session_state.image)
            st.session_state.result = (label, conf)

    with col2:
        if st.button("Reset System"):
            reset()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# RESULT SECTION
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("Diagnosis Result")

if st.session_state.result is not None:

    label, conf = st.session_state.result

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Prediction", label)

    with col2:
        st.metric("Confidence", f"{conf*100:.2f}%")

    st.progress(conf)

    if conf >= 0.8:
        color =  "#00c27a"
        st.success("High Confidence Prediction")
    elif conf >= 0.5:
        color = "#ffb300"
        st.warning("Medium Confidence")
    else:
        color = "#e53935"
        st.error("Low Confidence")

    st.markdown(f"""
    <style>
    .stProgress > div > div > div > div {{
        background: {color} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

else:
    st.info("Upload an image and click Run Analysis")


# FOOTER
st.markdown("---")
st.caption(" Deep Learning Oil Palm Disease Classification System")
