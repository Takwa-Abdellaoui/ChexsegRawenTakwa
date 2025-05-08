import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from model import ChexNetUNet
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F


class ChexpertDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir=None, transform=None, mask_size=(224, 224)):
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_size = mask_size

        self.pathologies = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')

        labels = torch.zeros(len(self.pathologies))
        for i, pathology in enumerate(self.pathologies):
            if pathology in self.data_frame.columns:
                value = self.data_frame.iloc[idx][pathology]
                if value == 1 or value == -1:
                    labels[i] = 1

        if self.transform:
            image = self.transform(image)

        masks = torch.zeros((len(self.pathologies), self.mask_size[0], self.mask_size[1]))
        if self.mask_dir:
            base_name = os.path.splitext(os.path.basename(self.data_frame.iloc[idx, 0]))[0]
            for i, pathology in enumerate(self.pathologies):
                mask_path = os.path.join(self.mask_dir, f"{base_name}_{pathology}.png")
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    mask = mask.resize(self.mask_size, Image.NEAREST)
                    mask = transforms.ToTensor()(mask)
                    masks[i] = mask

        return image, labels, masks


def train_model(model, train_loader, val_loader, criterion_cls, criterion_seg, optimizer, num_epochs=5, device='cuda'):
    since = time.time()
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'train_seg_loss': [], 'val_seg_loss': []
    }

    best_model_wts = model.state_dict()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else val_loader

            running_loss = 0.0
            running_cls_loss = 0.0
            running_seg_loss = 0.0

            with tqdm(dataloader, unit="batch") as tepoch:
                for inputs, labels, masks in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1} - {phase}")

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    masks = masks.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        pred_masks, pred_labels = model(inputs)

                        if pred_masks.shape != masks.shape:
                            pred_masks = F.interpolate(
                                pred_masks,
                                size=(masks.shape[2], masks.shape[3]),
                                mode='bilinear',
                                align_corners=False
                            )

                        cls_loss = criterion_cls(pred_labels, labels)
                        seg_loss = criterion_seg(pred_masks, masks)
                        loss = cls_loss + 0.5 * seg_loss

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    batch_size = inputs.size(0)
                    running_loss += loss.item() * batch_size
                    running_cls_loss += cls_loss.item() * batch_size
                    running_seg_loss += seg_loss.item() * batch_size

                    tepoch.set_postfix(loss=loss.item(), cls_loss=cls_loss.item(), seg_loss=seg_loss.item())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_cls_loss = running_cls_loss / len(dataloader.dataset)
            epoch_seg_loss = running_seg_loss / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}, CLS Loss: {epoch_cls_loss:.4f}, SEG Loss: {epoch_seg_loss:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_cls_loss'].append(epoch_cls_loss)
            history[f'{phase}_seg_loss'].append(epoch_seg_loss)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict().copy()

        print()

    time_elapsed = time.time() - since
    print(f'Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Meilleure perte de validation: {best_loss:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history


def plot_training_history(history, output_path='training_history.png'):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Perte Totale')
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['train_cls_loss'], label='Train')
    plt.plot(history['val_cls_loss'], label='Validation')
    plt.title('Perte de Classification')
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['train_seg_loss'], label='Train')
    plt.plot(history['val_seg_loss'], label='Validation')
    plt.title('Perte de Segmentation')
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    data_dir = "C:/Users/Admin/Desktop/CheXNet-master/chexseg/chexseg_data/images"
    image_dir = data_dir
    mask_dir = "C:/Users/Admin/Desktop/CheXNet-master/CheXNet_Masks"
    csv_file = "C:/Users/Admin/Desktop/CheXNet-master/chexseg/chexseg_data/labels.csv"
    output_dir = "output"
    batch_size = 8
    num_epochs = 5
    learning_rate = 1e-4
    image_size = (128, 128)
    mask_size = (128, 128)

    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ChexpertDataset(csv_file, image_dir, mask_dir, transform, mask_size=mask_size)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de {device}")

    model = ChexNetUNet(num_classes=14, num_masks=14).to(device)

    # ❄️ Geler les poids de l'encodeur CheXNet
    for param in model.encoder.features.parameters():
        param.requires_grad = False

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    criterion_cls = nn.BCELoss()
    criterion_seg = nn.BCELoss()
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    model, history = train_model(
        model, train_loader, val_loader,
        criterion_cls, criterion_seg, optimizer,
        num_epochs=num_epochs, device=device
    )

    torch.save(model.state_dict(), os.path.join(output_dir, "chexnet_unet_model.pth"))
    plot_training_history(history, os.path.join(output_dir, "training_history.png"))
    
if __name__ == "__main__":
    main()
