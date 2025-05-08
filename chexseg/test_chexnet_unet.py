import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import ChexNetUNet
import torchvision.models as models
from collections import OrderedDict
import cv2

# === CONFIGURATION ===
IMAGE_PATH = "chexseg_data/images/00000001_000.png"
UNET_MODEL_PATH = "output/chexnet_unet_model.pth"
CHEXNET_MODEL_PATH = "model.pth.tar"
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# === Prétraitement de l'image ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image_pil = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
base_img = np.array(image_pil.resize((224, 224))).astype(np.float32) / 255.0

# === 1. Chargement du modèle de segmentation (ChexNet + U-Net) ===
model = ChexNetUNet().to(DEVICE)
model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE))
model.eval()

# === 2. Chargement du modèle CheXNet uniquement pour classification ===
chexnet = models.densenet121(pretrained=False)
chexnet.classifier = torch.nn.Linear(chexnet.classifier.in_features, len(CLASS_NAMES))

checkpoint = torch.load(CHEXNET_MODEL_PATH, map_location=DEVICE)
if "state_dict" in checkpoint:
    print("✅ Chargement des poids CheXNet...")
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('densenet121.', '')
        new_state_dict[name] = v
    chexnet.load_state_dict(new_state_dict, strict=False)
else:
    chexnet = checkpoint
chexnet.eval()

# === Prédiction ===
with torch.no_grad():
    masks, _ = model(image_tensor)
    masks = masks.squeeze().cpu().numpy()   # (14, H, W)

    output_cls = chexnet(image_tensor)
    probs = torch.sigmoid(output_cls).squeeze().cpu().numpy()

# === Affichage Console des prédictions ===
positives = []
print("\n=== Résultats du modèle (CheXNet) ===")
for label, prob in zip(CLASS_NAMES, probs):
    if prob > THRESHOLD:
        print(f"{label:20} : {prob:.4f} (POSITIF)")
        positives.append((label, prob))
    else:
        print(f"{label:20} : {prob:.4f} (negatif)")

# === Affichage avec matplotlib ===
# === Affichage avec matplotlib ===
if positives:
    fig, axes = plt.subplots(1, len(positives), figsize=(5 * len(positives), 5))
    if len(positives) == 1:
        axes = [axes]

    for ax, (label, prob) in zip(axes, positives):
        idx = CLASS_NAMES.index(label)
        mask = masks[idx]

        # 🔄 Redimensionnement du masque à la taille de l’image
        resized_mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 🔥 Génération d'une heatmap colorée
        cmap = plt.get_cmap("plasma")
        heatmap = cmap(resized_mask ** 2)[:, :, :3]  # accentue les zones fortes


        # 🎨 Superposition avec l'image de base
        alpha = 0.6  # opacité plus forte
        overlay = base_img.copy()
        overlay[resized_mask > 0.3] = (1 - alpha) * overlay[resized_mask > 0.3] + alpha * heatmap[resized_mask > 0.3]


        ax.imshow(overlay)
        ax.set_title(f"{label}\nScore: {prob:.2f}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("✅ Aucune pathologie détectée avec score > 0.5.")
