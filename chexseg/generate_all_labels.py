import pandas as pd
import os

# === Chemins à adapter si besoin ===
data_entry_path = "C:/Users/Admin/Desktop/CheXNet-master/data/Data_Entry_2017_v2020.csv"
image_list_path = "C:/Users/Admin/Desktop/CheXNet-master/chexseg/chexseg_data/train_list.txt" # ou val_list.txt / test_list.txt
output_csv_path = "C:/Users/Admin/Desktop/CheXNet-master/chexseg/chexseg_data/labels.csv"

# Vérifier que les fichiers source existent
if not os.path.exists(data_entry_path):
    raise FileNotFoundError(f"Le fichier de données {data_entry_path} n'existe pas")
if not os.path.exists(image_list_path):
    raise FileNotFoundError(f"Le fichier de liste d'images {image_list_path} n'existe pas")

# S'assurer que le dossier de sortie existe
output_dir = os.path.dirname(output_csv_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Dossier créé: {output_dir}")

# === Lire les données ===
print("Lecture du fichier de données...")
df = pd.read_csv(data_entry_path)
df.columns = [col.strip() for col in df.columns]  # Nettoyage des noms de colonnes

# Assurer que 'Image Index' existe dans le dataframe
if 'Image Index' not in df.columns:
    possible_columns = [col for col in df.columns if 'image' in col.lower()]
    if possible_columns:
        print(f"'Image Index' non trouvé. Colonnes similaires trouvées: {possible_columns}")
    raise KeyError("La colonne 'Image Index' n'existe pas dans le fichier de données")

# Lire la liste des images concernées et extraire seulement le nom du fichier
print(f"Lecture de la liste d'images depuis {image_list_path}...")
image_names_with_annotations = []
image_names = []

with open(image_list_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            # Séparer le nom de fichier des annotations
            parts = line.split()
            if parts:
                image_name = parts[0]  # Prendre seulement le nom du fichier
                image_names.append(image_name)
                image_names_with_annotations.append(line)

print(f"Nombre d'images dans la liste: {len(image_names)}")

# Filtrer les lignes qui correspondent aux images dans la liste
filtered_df = df[df['Image Index'].isin(image_names)].copy()
print(f"Nombre d'images trouvées dans le dataset: {len(filtered_df)}")

if len(filtered_df) == 0:
    print("⚠️ Aucune image de la liste n'a été trouvée dans le dataset!")
    # Examiner quelques exemples pour diagnostiquer le problème
    print(f"Premiers noms d'images (extraits) dans la liste: {image_names[:5]}")
    print(f"Exemples d'Image Index dans le dataset: {df['Image Index'].head(5).tolist()}")
    
    # Essayer de faire la correspondance en ignorant la casse
    print("Tentative de correspondance en ignorant la casse...")
    image_names_lower = [name.lower() for name in image_names]
    df_image_index_lower = df['Image Index'].str.lower()
    matches = df[df_image_index_lower.isin(image_names_lower)].copy()
    if len(matches) > 0:
        print(f"Trouvé {len(matches)} correspondances en ignorant la casse!")
        filtered_df = matches
    else:
        print("Aucune correspondance trouvée même en ignorant la casse.")

# Si nous avons des images correspondantes, on continue
if len(filtered_df) > 0:
    # Transformer les étiquettes en colonnes binaires
    print("Création des colonnes de pathologies...")
    all_labels = set()
    for labels in filtered_df['Finding Labels']:
        if isinstance(labels, str):  # Vérifier que labels est une chaîne
            all_labels.update(labels.split('|'))
        else:
            print(f"⚠️ Valeur non-string trouvée dans 'Finding Labels': {labels}")

    print(f"Pathologies trouvées: {sorted(list(all_labels))}")

    # Créer des colonnes binaires pour chaque pathologie
    for label in all_labels:
        filtered_df[label] = filtered_df['Finding Labels'].apply(
            lambda x: 1 if isinstance(x, str) and label in x.split('|') else 0
        )

    # Garder uniquement le nom d'image et les colonnes de pathologies
    filtered_df.rename(columns={'Image Index': 'image_name'}, inplace=True)
    final_df = filtered_df[['image_name'] + sorted(list(all_labels))]
else:
    # Si on n'a toujours pas de correspondance, on va créer le fichier labels.csv à partir des annotations
    print("Création de labels.csv à partir des annotations dans le fichier de liste...")
    
    # Analyser les annotations pour déterminer les colonnes
    sample_line = image_names_with_annotations[0] if image_names_with_annotations else ""
    parts = sample_line.split()
    
    if len(parts) > 1:  # S'assurer qu'il y a des annotations
        # Déterminer le nombre de catégories
        num_categories = len(parts) - 1  # nombre total moins le nom du fichier
        
        # Créer un DataFrame vide avec les colonnes appropriées
        columns = ['image_name'] + [f'category_{i}' for i in range(num_categories)]
        data = []
        
        for line in image_names_with_annotations:
            parts = line.split()
            if len(parts) >= num_categories + 1:
                row = [parts[0]] + [int(parts[i+1]) for i in range(num_categories)]
                data.append(row)
        
        final_df = pd.DataFrame(data, columns=columns)
        print(f"DataFrame créé à partir des annotations avec {len(final_df)} images et {num_categories} catégories")
    else:
        print("Impossible de créer le fichier - format de ligne inattendu")
        final_df = pd.DataFrame(columns=['image_name'])

# Sauvegarder le fichier CSV
print(f"Sauvegarde des données vers {output_csv_path}...")
final_df.to_csv(output_csv_path, index=False)

# Afficher un résumé des données
print("\n=== Résumé ===")
print(f"Nombre total d'images traitées: {len(final_df)}")
print(f"Nombre de colonnes: {len(final_df.columns) - 1}")  # moins la colonne image_name
print("Aperçu des données:")
print(final_df.head())

print(f"\n✅ labels.csv généré avec succès à : {output_csv_path}")
print(f"   Dimensions du fichier: {final_df.shape[0]} lignes × {final_df.shape[1]} colonnes")