
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models


MODEL_PATH = Path(__file__).with_name("resnet18_fever_detector.pth") 
TARGET_SIZE = (224, 224)
ROI_CFG = {  
    "head": (0.00, 0.20),  
    "udder": (0.70, 1.00),  
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """Load fine‑tuned ResNet‑18."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.25), torch.nn.Linear(model.fc.in_features, 1)
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval().to(DEVICE)
    return model


def load_resize(fp, size=TARGET_SIZE):
    img = Image.open(fp).convert("RGB")
    return np.array(img.resize(size, resample=Image.BILINEAR))


def thermal_to_mask(img_rgb: np.ndarray):
    """Very fast silhouette estimator on FLIR colormapped frames."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = hsv[..., 2]  
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(mask)
    big = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [big], -1, 255, -1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=3)
    return clean  

def overlay_roi(img_rgb: np.ndarray, mask: np.ndarray, roi_cfg: dict = ROI_CFG):
    """Return copy with ROI rectangles drawn."""
    out = img_rgb.copy()
    h, w = mask.shape
    for name, (y0, y1) in roi_cfg.items():
        p0, p1 = (0, int(y0 * h)), (w, int(y1 * h))
        color = (0, 255, 0) if name == "head" else (255, 0, 0)
        cv2.rectangle(out, p0, p1, color, 2)
        cv2.putText(out, name.upper(), (p0[0] + 5, p0[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def predict(img_rgb: np.ndarray, model):
    tfm = T.Compose([
        T.ToTensor(),
        T.Resize(TARGET_SIZE),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    x = tfm(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    return prob  


def grad_cam_vis(img_rgb: np.ndarray, model):
    tfm = T.Compose([T.ToTensor(), T.Resize(TARGET_SIZE)])
    tensor = tfm(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=torch.cuda.is_available())
    mask = cam(tensor)[0]
    vis = show_cam_on_image(img_rgb / 255.0, mask, use_rgb=True)
    return vis


st.set_page_config("Holstein Fever Early‑Warning Demo", layout="centered")
st.title("🐄 Holstein Cattle – Thermal Early Disease Demo")

st.markdown(
    "Upload a **thermal JPEG** from the *Holstein Cattle Recognition* dataset (or one with similar FLIR colormap).\n"
    "The app will: 1️⃣ segment the cow, 2️⃣ draw ROI boxes, 3️⃣ run the CNN fever detector,"
    " 4️⃣ display Grad‑CAM heatmap explaining the decision."
)

uploaded = st.file_uploader("Choose thermal image …", type=["jpg", "jpeg", "png"])


if MODEL_PATH.exists():
    model = load_model(MODEL_PATH)
else:
    st.warning("`model.pth` not found – predictions disabled.")
    model = None

if uploaded is not None:
   
    img_rgb = load_resize(uploaded)
    mask = thermal_to_mask(img_rgb)
    overlay = overlay_roi(img_rgb, mask)

   
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Thermal + ROI")
        st.image(overlay, clamp=True, channels="RGB")
    with col2:
        st.subheader("Silhouette mask")
        st.image(mask, clamp=True)

    if model is not None:
        prob = predict(img_rgb, model)
        fever = prob > 0.5
        st.success(f"**Fever probability:** {prob:.2%} → {'🚨 Likely' if fever else '✅ Normal'}")

        # Grad‑CAM
        with st.spinner("Computing Grad‑CAM …"):
            vis = grad_cam_vis(img_rgb, model)
        st.subheader("Model attention (Grad‑CAM)")
        st.image(vis, clamp=True, channels="RGB")
    
    st.caption("Note: This is a *demo* using an uncalibrated brightness proxy, not absolute temperature.")
else:
    st.info("Upload an image to get started.")


st.sidebar.header("About this demo")
st.sidebar.markdown(
    "* **Dataset**: [Holstein Cattle Recognition](https://doi.org/10.34894/O1ZBSA) – University of Groningen.\n"
    "* **Model**: ResNet‑18 fine‑tuned to classify *fever* vs *normal* based on ROI brightness proxy.\n"
    "* **Paper idea**: Early detection of mastitis / fever using thermal cameras in barns."
)

