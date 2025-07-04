# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Pvyr90suKViIAZCU_6s5agR0QsIHxVS9
"""

from google.colab import files

uploaded = files.upload()

import zipfile
import os

zip_path = 'dataverse_files.zip'
extract_path = 'holstein_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Extracted to:", extract_path)

search_root = 'holstein_data'

image_paths = []
valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

for root, _, files in os.walk(search_root):
    for file in files:
        if file.lower().endswith(valid_exts):
            full_path = os.path.join(root, file)
            image_paths.append(full_path)

print(f"✅ Found {len(image_paths)} images.")

import os

for root, dirs, files in os.walk('holstein_data'):
    print("📁", root)
    break

print("Folders inside 'holstein_data':", os.listdir("holstein_data"))

import zipfile
import os

subset_dir = "holstein_data"
for filename in os.listdir(subset_dir):
    if filename.endswith(".zip"):
        zip_path = os.path.join(subset_dir, filename)
        extract_path = os.path.join(subset_dir, filename.replace(".zip", ""))
        print(f"📦 Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

print("✅ All subset zip files extracted.")

search_root = 'holstein_data'

image_paths = []
valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

for root, _, files in os.walk(search_root):
    for file in files:
        if file.lower().endswith(valid_exts):
            full_path = os.path.join(root, file)
            image_paths.append(full_path)

print(f"✅ Found {len(image_paths)} images.")

import pandas as pd
def find_pairs(root):
    records = []
    for dirpath, dirnames, filenames in os.walk(root):

        all_imgs = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith((".jpg", ".jpeg"))]
        thermals = sorted([f for f in all_imgs if "thermal" in f.lower()])
        visibles = sorted([f for f in all_imgs if "visible" in f.lower()])
        if len(thermals) != len(visibles) or len(thermals) == 0:
            continue
        for t_path, v_path in zip(thermals, visibles):
            cow_id = os.path.basename(os.path.dirname(t_path))  # e.g., FLIR1234
            records.append({
                "cow_id": cow_id,
                "thermal": t_path,
                "visible": v_path
            })
    return pd.DataFrame(records)

df_pairs = find_pairs("holstein_data")
print(f"✅ Total image pairs found: {len(df_pairs)}")
df_pairs.head()

import random
random.shuffle(image_paths)

for path in image_paths[:10]:
    print(path)

import os
import pandas as pd
from collections import defaultdict

def find_pairs_by_id(root):
    thermal_map = {}
    visible_map = {}

    for dirpath, _, filenames in os.walk(root):
        if '__MACOSX' in dirpath:
            continue
        for f in filenames:
            if not f.lower().endswith((".jpg", ".jpeg")):
                continue
            if not f.lower().startswith("flir"):
                continue
            img_id = f.split('-')[0].split('.')[0].strip()
            full_path = os.path.join(dirpath, f)
            if "full" in f.lower():
                visible_map[img_id] = full_path
            else:
                thermal_map[img_id] = full_path


    common_ids = sorted(set(thermal_map.keys()) & set(visible_map.keys()))
    print(f"Found {len(common_ids)} paired IDs")

    records = []
    for img_id in common_ids:
        records.append({
            "cow_id": img_id,
            "thermal": thermal_map[img_id],
            "visible": visible_map[img_id]
        })

    return pd.DataFrame(records)


df_pairs = find_pairs_by_id("holstein_data")
print(f"✅ Total matched image pairs: {len(df_pairs)}")
df_pairs.head()

TARGET_SIZE = (224, 224)
ROI_CFG = {
    "head": (0.00, 0.20),
    "udder": (0.70, 1.00),
}


import numpy as np
from PIL import Image
import cv2

def load_resize(fp, size=TARGET_SIZE):
    img = Image.open(fp).convert("RGB")
    img = img.resize(size, resample=Image.BILINEAR)
    return np.array(img)

def thermal_to_mask(img_rgb):
    hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    gray = hsv[..., 2]
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return np.zeros_like(mask)

    big = max(cnts, key=cv2.contourArea)

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [big], -1, 255, -1)

    kernel = np.ones((9, 9), np.uint8)
    clean  = cv2.morphologyEx(
        clean, cv2.MORPH_CLOSE, kernel, iterations=3)

    return clean

def mean_temp_in_roi(thermal_rgb, mask, roi_range):
    """
    Estimate mean pixel intensity (proxy for temperature) in a vertical slice of the mask.
    """
    h = mask.shape[0]
    y0, y1 = int(h * roi_range[0]), int(h * roi_range[1])


    roi_mask = np.zeros_like(mask)
    roi_mask[y0:y1, :] = mask[y0:y1, :]
    idx = roi_mask > 0


    hsv = cv2.cvtColor(thermal_rgb, cv2.COLOR_RGB2HSV)
    values = hsv[:, :, 2][idx]

    if len(values) == 0:
        return np.nan
    return values.mean()

records = []

for i, row in df_pairs.iterrows():
    ther = load_resize(row['thermal'])
    mask = thermal_to_mask(ther)

    roi_stats = {}
    for roi_name, roi_range in ROI_CFG.items():
        roi_stats[f"{roi_name}_meanV"] = mean_temp_in_roi(ther, mask, roi_range)

    record = {
        "cow_id": row['cow_id'],
        "thermal_path": row['thermal'],
        "visible_path": row['visible'],
        **roi_stats
    }
    records.append(record)

df_temps = pd.DataFrame(records)
print("✅ Done. Shape:", df_temps.shape)
df_temps.head()

df_temps = df_temps.dropna(subset=['head_meanV', 'udder_meanV']).reset_index(drop=True)
print("After dropping NaNs:", df_temps.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(df_temps['head_meanV'], df_temps['udder_meanV'], alpha=0.5)
plt.xlabel('Head Brightness (meanV)')
plt.ylabel('Udder Brightness (meanV)')
plt.title('Head vs. Udder Brightness')
plt.grid(True)
plt.show()

mu = df_temps['udder_meanV'].mean()
sigma = df_temps['udder_meanV'].std()


df_temps['fever_flag'] = df_temps['udder_meanV'] > (mu + 2 * sigma)

print("⚠️ Fever cases:", df_temps['fever_flag'].sum())
df_temps[df_temps['fever_flag']].head()

import matplotlib.pyplot as plt
import random


sample = df_temps.sample(1).iloc[0]


thermal = load_resize(sample['thermal_path'])
mask = thermal_to_mask(thermal)


plt.figure(figsize=(6, 6))
plt.imshow(thermal)
plt.imshow(mask, cmap='gray', alpha=0.3)
plt.title(f"Thermal with ROI Mask - Cow {sample['cow_id']} - Fever: {sample['fever_flag']}")
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt
import cv2

def plot_roi_visual(sample):
    thermal = load_resize(sample['thermal_path'])
    mask = thermal_to_mask(thermal)

    h, w = mask.shape
    head_box = [0, 0, w, int(0.2 * h)]
    udder_box = [0, int(0.7 * h), w, h]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))


    axes[0].imshow(thermal)
    axes[0].set_title("Thermal Image")
    axes[0].axis('off')


    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Silhouette Mask")
    axes[1].axis('off')


    axes[2].imshow(thermal)
    axes[2].imshow(mask, cmap='gray', alpha=0.3)


    for box, label, color in zip([head_box, udder_box], ['Head ROI', 'Udder ROI'], ['lime', 'red']):
        x, y, x2, y2 = box[0], box[1], box[2], box[3]
        axes[2].add_patch(plt.Rectangle((x, y), x2 - x, y2 - y, edgecolor=color, facecolor='none', lw=2))
        axes[2].text(x + 5, y + 15, label, color=color, fontsize=10, weight='bold')


    fever_text = f"Fever: {sample['fever_flag']}"
    axes[2].set_title(f"Overlay + ROI Boxes — {sample['cow_id']} — {fever_text}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


for _ in range(5):
    sample = df_temps.sample(1).iloc[0]
    plot_roi_visual(sample)

import seaborn as sns, matplotlib.pyplot as plt


viz_df = df_temps.dropna(subset=['head_meanV','udder_meanV']).copy()

viz_df['status'] = viz_df['fever_flag'].map({True:'Fever', False:'Normal'})

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    data=viz_df.melt(
        id_vars=['status'],
        value_vars=['head_meanV','udder_meanV'],
        var_name='ROI', value_name='meanV'),
    x='ROI', y='meanV', hue='status', ax=ax, errorbar='sd')

ax.set_title('Mean V-channel brightness (proxy °C) per ROI')
ax.set_ylabel('HSV-V  (0–255)')
plt.tight_layout(); plt.show()

!pip install -q torch torchvision timm torchmetrics grad-cam==1.4.6
import torch, torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import pandas as pd, numpy as np, cv2, matplotlib.pyplot as plt

MEAN = [0.5]*3 ; STD = [0.5]*3

tfm_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(MEAN, STD)
])
tfm_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(MEAN, STD)
])

class HolsteinThermal(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self):  return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["thermal_path"]).convert("RGB")
        x     = self.tfm(img)
        y     = torch.tensor(row["label"], dtype=torch.float32)
        return x, y

df = df_temps.copy()
df["label"] = df["fever_flag"].astype(int)
print(df["label"].value_counts())

train_df = df.sample(frac=0.8, random_state=0)
val_df   = df.drop(train_df.index)

train_dl = DataLoader(HolsteinThermal(train_df, tfm_train),
                      batch_size=32, shuffle=True,  num_workers=2)
val_dl   = DataLoader(HolsteinThermal(val_df,   tfm_val),
                      batch_size=32, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(model.fc.in_features, 1)
)
model = model.to(device)

pos_weight = torch.tensor(
    [(len(train_df) - train_df["label"].sum()) / train_df["label"].sum()]
).to(device)          # << use device, not "cuda"

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimiser = optim.Adam(model.parameters(), lr=3e-4)

def run_epoch(dl, train):
    model.train(train)
    loss_sum, n = 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        if train:
            optimiser.zero_grad()
        y_hat = model(x)
        loss  = criterion(y_hat, y)
        if train:
            loss.backward(); optimiser.step()
        loss_sum += loss.item()*len(y); n += len(y)
    return loss_sum/n

for epoch in range(10):
    tl = run_epoch(train_dl, True)
    vl = run_epoch(val_dl,   False)
    print(f"epoch {epoch:02d} | train {tl:.3f} | val {vl:.3f}")

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)  # CPU

def show_grad_cam(row):
    img = Image.open(row["thermal_path"]).convert("RGB")
    rgb = np.array(img.resize((224, 224))) / 255.0
    tensor = tfm_val(img).unsqueeze(0).to(device)

    mask = cam(tensor)[0]
    vis = show_cam_on_image(rgb, mask, use_rgb=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(vis)
    plt.title(f"FLIR {row['cow_id']}  Fever? {bool(row['label'])}")
    plt.axis("off")
    plt.show()


show_grad_cam(train_df[train_df["label"] == 0].iloc[0])
show_grad_cam(train_df[train_df["label"] == 1].iloc[0])

from sklearn.metrics import classification_report, confusion_matrix


y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for xb, yb in val_dl:
        xb = xb.to(device)
        preds = model(xb).squeeze(1)
        y_true.extend(yb.cpu().numpy())
        y_pred.extend((torch.sigmoid(preds) > 0.5).int().cpu().numpy())


print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=3))

print(classification_report(y_true, y_pred, digits=3))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Fever"],
            yticklabels=["Normal","Fever"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.show()

torch.save(model.state_dict(), "resnet18_fever_detector.pth")

def predict_single_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = tfm_val(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).item()
    return "Fever" if pred > 0.5 else "Normal", pred