
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2


class FeverResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(FeverResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_model():
    model = FeverResNet18()
    model.load_state_dict(torch.load("resnet18_fever_detector.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        return probs[0][1].item()  # probability of class 1 (fever)


def generate_grad_cam(input_image, model):
    model.eval()
    image_tensor = transform(input_image).unsqueeze(0).requires_grad_()
    output = model(image_tensor)

    target_class = output.argmax().item()
    output[0, target_class].backward()

    gradients = model.model.layer4[1].conv2.weight.grad
    activations = model.model.layer4[1].conv2.weight

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[0, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap

st.title("🐄 AI-Powered Fever Detection in Livestock (Demo)")
st.write("Upload a thermal image of a cow to detect fever signs using ResNet18 + Grad-CAM")

uploaded_file = st.file_uploader("Choose a thermal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

   
    prob = predict(image, model)
    fever = prob > 0.5
    st.subheader("Prediction Result")
    st.write(f"**Fever Probability:** {prob:.2f}")
    if fever:
        st.error("🚨 Fever likely!")
    else:
        st.success("✅ Normal")

    
    st.subheader("Grad-CAM Explanation")
    cam = generate_grad_cam(image, model)
    img_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(img_np) / 255
    overlay = overlay / np.max(overlay)
    st.image(np.uint8(255 * overlay), caption="Grad-CAM", use_column_width=True)

st.sidebar.header("About this demo")
st.sidebar.markdown("""
* **Model:** ResNet18 fine-tuned to classify *fever* vs *normal*.
* **Dataset:** Holstein Cattle (thermal images)
* **Demo Goal:** Show explainable AI (Grad-CAM) + CV for livestock fever detection.
""")

