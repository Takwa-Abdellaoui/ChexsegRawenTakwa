import os
import io
import base64
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # backend non-GUI
import matplotlib.pyplot as plt
from collections import OrderedDict
import random

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
def executer_analyse(image_path):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    MODEL_PATH = "model.pth.tar"
    CAM_DIR = "chexseg_data/generated_cams"
    THRESHOLD = 0.5

    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASS_NAMES))

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('densenet121.', '')
        name = name.replace('norm.1', 'norm1').replace('norm.2', 'norm2')
        name = name.replace('conv.1', 'conv1').replace('conv.2', 'conv2')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    image_pil = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
    probs = torch.sigmoid(output).numpy()[0]

    base_img = np.array(image_pil.resize((224, 224))).astype(np.float32) / 255.0
    results = []

    for label, prob in zip(CLASS_NAMES, probs):
        if prob > THRESHOLD:
            cam_path = os.path.join(CAM_DIR, f"{os.path.splitext(os.path.basename(image_path))[0]}_{label}.png")
            if os.path.exists(cam_path):
                cam = Image.open(cam_path).resize((224, 224)).convert("RGB")
                cam_np = np.array(cam).astype(np.float32) / 255.0
                overlay = np.clip(0.6 * base_img + 0.4 * cam_np, 0, 1)

                fig, ax = plt.subplots()
                ax.imshow(overlay)
                ax.axis("off")
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                cam_base64 = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close(fig)

                results.append({
                    "label": label,
                    "score": round(float(prob), 4),
                    "cam": cam_base64
                })

    # === Courbe matplotlib ===
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(CLASS_NAMES, probs, color=["#2C3E50" if p > THRESHOLD else "gray" for p in probs])
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax2.set_ylabel("Probabilité")
    ax2.set_title("Prédictions du modèle")
    fig2.tight_layout()

    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    buf2.seek(0)
    chart_base64 = base64.b64encode(buf2.read()).decode("utf-8")
    buf2.close()
    plt.close(fig2)

    return results, chart_base64
