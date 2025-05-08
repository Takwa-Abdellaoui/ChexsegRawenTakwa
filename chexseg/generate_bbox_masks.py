import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw

# 📁 Dossiers
bbox_csv = "BBox_List_2017.csv"
image_dir = "chexseg_data/images"
output_mask_dir = "real_bbox_masks"
os.makedirs(output_mask_dir, exist_ok=True)

# 📋 Pathologies
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
label_to_index = {label: i for i, label in enumerate(CLASS_NAMES)}

# 📄 Chargement et nettoyage des colonnes
bbox_df = pd.read_csv(bbox_csv)
bbox_df.columns = ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h', 'drop1', 'drop2', 'drop3']

# 🖼️ Génération des masques
for _, row in bbox_df.iterrows():
    img_name = row['Image Index']
    pathology = row['Finding Label']

    if pathology not in label_to_index:
        continue

    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        continue

    image = Image.open(img_path).convert("L")
    w_img, h_img = image.size
    mask = Image.new("L", (w_img, h_img), 0)
    draw = ImageDraw.Draw(mask)

    x, y, bw, bh = row['x'], row['y'], row['w'], row['h']
    box = [x, y, x + bw, y + bh]
    draw.rectangle(box, fill=255)

    base = os.path.splitext(img_name)[0]
    mask_output_path = os.path.join(output_mask_dir, f"{base}_{pathology}.png")
    mask.save(mask_output_path)

print("✅ Masques générés avec succès dans :", output_mask_dir)
