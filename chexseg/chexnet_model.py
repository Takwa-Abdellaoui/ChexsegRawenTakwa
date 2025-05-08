# encoding: utf-8
"""
The main CheXNet model implementation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score

# Define paths
CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

DATA_DIR = os.path.abspath("C:/Users/Admin/Downloads/Chest-Xray14/images/images")
TEST_IMAGE_LIST = "C:/Users/Admin/Downloads/CheXNet-master/ChestX-ray14/labels/test_list.txt"
BATCH_SIZE = 16  

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    torch.cuda.empty_cache()  # Libérer la mémoire GPU inutilisée

    # Initialisation et chargement du modèle
    # Initialisation et chargement du modèle
    model = DenseNet121(N_CLASSES).to(device).half()  # Convertir en 16-bit

    if os.path.isfile(CKPT_PATH):
        print("=> Loading checkpoint...")
        checkpoint = torch.load(CKPT_PATH)

        # Adapter automatiquement selon que le modèle a été enregistré avec DataParallel ou non
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            # Si les clés ont le préfixe "module.", on le retire
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print("=> Checkpoint loaded successfully.")
    else:
        print("=> No checkpoint found.")

    # Mettre en DataParallel **après** avoir chargé les poids
    model = torch.nn.DataParallel(model).to(device)


    if os.path.isfile(CKPT_PATH):
        print("=> Loading checkpoint...")
        checkpoint = torch.load(CKPT_PATH)
        model.module.load_state_dict(checkpoint['state_dict'])

        print("=> Checkpoint loaded successfully.")
    else:
        print("=> No checkpoint found.")

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Préparation du dataset et dataloader (simplification des transformations)
    test_dataset = ChestXrayDataSet(
        data_dir=DATA_DIR,
        image_list_file=TEST_IMAGE_LIST,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # Remplace TenCrop pour économiser de la mémoire
            transforms.ToTensor(),
            normalize
        ])
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)

    # Passage en mode évaluation
    model.eval()

    with torch.no_grad():  # Évite l'utilisation inutile de la mémoire GPU
        for i, (inp, target) in enumerate(test_loader):
            target = target.to(device)
            gt = torch.cat((gt, target), 0)

            input_var = inp.to(device).half()  # Conversion en 16-bit
            output = model(input_var)
            pred = torch.cat((pred, output.detach()), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print(f'The average AUROC is {AUROC_avg:.3f}')
    for i in range(N_CLASSES):
        print(f'The AUROC of {CLASS_NAMES[i]} is {AUROCs[i]:.3f}')

def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores."""
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

class DenseNet121(nn.Module):
    """Modified DenseNet121 model for CheXNet."""

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)

if __name__ == '__main__':
    main()