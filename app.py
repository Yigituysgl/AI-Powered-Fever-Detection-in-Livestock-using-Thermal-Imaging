import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr


class FeverResNet(nn.Module):
    def __init__(self):
        super(FeverResNet, self).__init__()
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeverResNet()
model.load_state_dict(torch.load("resnet18_fever_detector.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = torch.sigmoid(output).item()
        label = "Fever" if pred > 0.5 else "Normal"
        confidence = round(pred if pred > 0.5 else 1 - pred, 3)
    return f"{label} (Confidence: {confidence})"


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Cow Fever Detector",
    description="Upload a thermal image of a cow to detect signs of fever."
)

iface.launch()

