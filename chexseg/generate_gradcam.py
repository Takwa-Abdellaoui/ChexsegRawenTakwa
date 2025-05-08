import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# === CONFIGURATION ===
IMAGE_PATH = "chexseg_data/images/00000001_000.png"
MODEL_PATH = "model.pth.tar"
OUTPUT_DIR = "gradcam_outputs"
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load Image ===
image_pil = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
base_img = np.array(image_pil.resize((224, 224))).astype(np.float32) / 255.0

# === Load CheXNet Model ===
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if "state_dict" in checkpoint:
    print("Chargement des poids du modèle...")
    new_state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        name = k.replace("module.", "").replace("densenet121.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
else:
    model = checkpoint

model.to(DEVICE)
model.eval()

# === Prédictions ===
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.sigmoid(outputs)[0].cpu().numpy()

# === Sélection des pathologies POSITIVES ===
positives = [(i, CLASS_NAMES[i], prob) for i, prob in enumerate(probs) if prob > THRESHOLD]

if not positives:
    print("Aucune pathologie détectée avec score > 0.5")
    exit()

# === GradCAM ===
target_layer = model.features[-1]
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

# === Création du dossier de sortie ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Génération des CAMs ===
fig, axes = plt.subplots(1, len(positives), figsize=(5 * len(positives), 5))
if len(positives) == 1:
    axes = [axes]

for ax, (idx, label, prob) in zip(axes, positives):
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(idx)])[0]
    cam_image = show_cam_on_image(base_img, grayscale_cam, use_rgb=True)
    ax.imshow(cam_image)
    ax.set_title(f"{label}\nScore: {prob:.2f}")
    ax.axis("off")

    # Sauvegarde optionnelle
    out_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(IMAGE_PATH)[:-4]}_{label}.png")
    Image.fromarray(cam_image).save(out_path)

plt.tight_layout()
plt.show()
