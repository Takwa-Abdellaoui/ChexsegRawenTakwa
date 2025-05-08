import os
import pandas as pd

# === Chemins à adapter ===
image_folder = "C:/Users/Admin/Desktop/CheXNet-master/chexseg/chexseg_data/images"
data_entry_csv = "C:/Users/Admin/Desktop/CheXNet-master/chexseg/chexseg_data/Data_Entry_2017_v2020.csv"
output_txt = "C:/Users/Admin/Desktop/CheXNet-master/chexseg/chexseg_data/train_list.txt"

# === Les 14 pathologies de ChestX-ray14 ===
pathologies = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# === Lire le CSV ===
df = pd.read_csv(data_entry_csv)
df.columns = [c.strip() for c in df.columns]
df['Finding Labels'] = df['Finding Labels'].fillna('No Finding')

# Lister les images présentes physiquement
image_set = set(os.listdir(image_folder))

# Ouvrir le fichier de sortie
with open(output_txt, 'w') as f:
    for _, row in df.iterrows():
        image_name = row['Image Index']
        if image_name not in image_set:
            continue  # Ignorer les images non présentes physiquement

        labels = row['Finding Labels'].split('|')
        binary_labels = ['1' if p in labels else '0' for p in pathologies]
        line = f"{image_name} {' '.join(binary_labels)}\n"
        f.write(line)

print(f"✅ Fichier train_list.txt généré avec succès à : {output_txt}")
