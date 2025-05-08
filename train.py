import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from read_data import ChestXrayDataSet  # Utilisez le bon nom de classe
import os  # Ajoutez cette ligne pour utiliser os.path

# Définir le modèle (DenseNet121)
class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Hyperparamètres
N_CLASSES = 14  # Nombre de classes dans ChestX-ray14
BATCH_SIZE = 32  # Taille du lot
EPOCHS = 10  # Nombre d'époques
LEARNING_RATE = 0.001  # Taux d'apprentissage

# Chemins des données
DATA_DIR = os.path.join("C:", "Users", "Admin", "Downloads", "CheXNet-master", "data")  
IMAGE_LIST_FILE = r"C:\Users\Admin\Downloads\CheXNet-master\ChestX-ray14\labels\train_list.txt"  

# Vérifier que le fichier existe
if not os.path.exists(IMAGE_LIST_FILE):
    raise FileNotFoundError(f"Le fichier {IMAGE_LIST_FILE} n'existe pas. Veuillez vérifier le chemin.")

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner les images à 224x224
    transforms.ToTensor(),  # Convertir en tenseur
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalisation
])

# Charger les données
train_dataset = ChestXrayDataSet(data_dir=DATA_DIR, image_list_file=IMAGE_LIST_FILE, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialiser le modèle, la fonction de perte et l'optimiseur
model = DenseNet121(N_CLASSES)
criterion = nn.BCEWithLogitsLoss()  # Fonction de perte pour la classification multi-labels
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Boucle d'entraînement
for epoch in range(EPOCHS):
    model.train()  # Passer le modèle en mode entraînement
    running_loss = 0.0

    for inputs, labels in train_loader:
        # Remettre à zéro les gradients
        optimizer.zero_grad()

        # Passer les données dans le modèle
        outputs = model(inputs)

        # Calculer la perte
        loss = criterion(outputs, labels)

        # Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()

        # Accumuler la perte
        running_loss += loss.item()

    # Afficher la perte moyenne pour l'époque
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}")

# Sauvegarder le modèle
torch.save(model.state_dict(), "model.pth")
print("Modèle entraîné et sauvegardé !")