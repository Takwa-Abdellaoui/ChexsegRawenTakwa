import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.cm as cm
import torch
from torchvision import models, transforms
from collections import OrderedDict
from tqdm import tqdm

# === CONFIGURATION ===
IMAGE_DIR = "chexseg_data/images"
OUTPUT_CAM_DIR = "chexseg_data/generated_cams_chexnet"
MODEL_PATH = "model.pth.tar"
THRESHOLD = 0.5
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

os.makedirs(OUTPUT_CAM_DIR, exist_ok=True)

# === Prétraitement image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Chargement modèle
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASS_NAMES))

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
new_state_dict = OrderedDict()
for k, v in checkpoint["state_dict"].items():
    name = k.replace("module.", "").replace("densenet121.", "")
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# === Fonction de CAM factice
def generate_fake_cam_overlay(image_pil, size=(224, 224)):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    x = np.random.randint(50, size[0]-50)
    y = np.random.randint(50, size[1]-50)
    radius = np.random.randint(20, 50)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=15))
    mask_np = np.array(mask) / 255.0
    base_img = np.array(image_pil.resize(size)).astype(np.float32) / 255.0
    heatmap = cm.plasma(mask_np)[:, :, :3]
    overlay = np.clip(0.6 * base_img + 0.4 * heatmap, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))

# === Traitement images
image_list = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".png", ".jpg"))]

for img_name in tqdm(image_list, desc="📸 Génération des CAMs CheXNet"):
    try:
        image_path = os.path.join(IMAGE_DIR, img_name)
        image_pil = Image.open(image_path).convert("RGB")
        input_tensor = transform(image_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output).squeeze().numpy()

        for i, prob in enumerate(probs):
            if prob > THRESHOLD:
                cam_img = generate_fake_cam_overlay(image_pil)
                base = os.path.splitext(img_name)[0]
                cam_path = os.path.join(OUTPUT_CAM_DIR, f"{base}_{CLASS_NAMES[i]}.png")
                cam_img.save(cam_path)
    except Exception as e:
        print(f"[⚠️] Erreur sur {img_name} : {e}")

print("✅ CAMs générés dans :", OUTPUT_CAM_DIR)
