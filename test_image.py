import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import numpy as np
import random
import torch
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = "model.pth.tar"  
try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
except FileNotFoundError:
    print(f"Erreur : le fichier {model_path} est introuvable.")
    exit()

# Initialiser le modèle
model = models.densenet121(pretrained=False)
num_classes = 14  # Correspond à votre liste de labels
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# Vérifier si le fichier contient un état de modèle (state_dict) ou un modèle complet
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        print("Chargement des poids du modèle...")
        state_dict = checkpoint["state_dict"]
        
        # Nettoyage des clés du state_dict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # Supprimer 'module.' si présent
            name = name.replace('densenet121.', '')  # Supprimer 'densenet121.' si présent
            # Corriger les noms de couches si nécessaire
            name = name.replace('norm.1', 'norm1')
            name = name.replace('norm.2', 'norm2')
            name = name.replace('conv.1', 'conv1')
            name = name.replace('conv.2', 'conv2')
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("Chargement du modèle complet...")
        model = checkpoint  # Si le modèle a été sauvegardé en entier
else:
    print("Format de checkpoint non reconnu")
    exit()

model.eval()

# Liste des maladies prédictibles
labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
          "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
          "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

# Charger et prétraiter l'image
image_path = "data/images/00000001_002.png"  # Mets ici le chemin correct de ton image
try:
    image = Image.open(image_path).convert('RGB')
except FileNotFoundError:
    print(f"Erreur : l'image {image_path} est introuvable.")
    exit()

# Transformation adaptée à DenseNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = transform(image).unsqueeze(0)

# Vérifier si l'image et le modèle sont bien chargés
if image is None or model is None:
    print("Erreur : chargement de l'image ou du modèle échoué.")
    exit()

# Faire la prédiction
with torch.no_grad():
    output = model(image)
probs = torch.sigmoid(output).numpy()[0]

# Afficher les résultats
print("\n=== Résultats du modèle ===")
for i, (label, prob) in enumerate(zip(labels, probs)):
    print(f"{label:20} : {prob:.4f} {'(POSITIF)' if prob > 0.5 else '(negatif)'}")