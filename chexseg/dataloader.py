from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, mask_file, transform=None):
        self.image_dir = image_dir
        self.masks = np.load(mask_file, allow_pickle=True)  # (4999, 14, 224, 224)
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))  # suppose que l’ordre correspond aux masques

        assert len(self.image_files) == self.masks.shape[0], "Le nombre d'images et de masques ne correspond pas."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("L")  # images en niveaux de gris

        mask = self.masks[idx]  # shape : (14, 224, 224)

        if self.transform:
            image = self.transform(image)

        return image, mask
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Convertir l'image en tensor
])
dataset = ChestXrayDataset(image_dir='chexseg_data/images', mask_file='masks.npy', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
