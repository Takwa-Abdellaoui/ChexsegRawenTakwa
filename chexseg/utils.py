import torch
import matplotlib.pyplot as plt
import numpy as np

# Liste des 14 pathologies du ChestX-ray14 (optionnel pour affichage)
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

def visualize_all_classes(model, dataloader, device='cpu', threshold=0.5, max_images=5):
    """
    Affiche les prédictions du modèle pour les 14 classes sur un nombre limité d'images.
    
    Args:
        model: le modèle ChexNet-UNet entraîné
        dataloader: DataLoader contenant les images et masques
        device: 'cpu' ou 'cuda'
        threshold: seuil pour binariser les masques prédits
        max_images: nombre d’images à afficher
    """
    model.eval()
    count = 0

    with torch.no_grad():
        for images, true_masks, labels in dataloader:
            if count >= max_images:
                break

            images = images.to(device)
            outputs = model(images)
            preds = (outputs > threshold).float().cpu().numpy()  # Binarisation

            image = images[0].cpu().squeeze(0).numpy()  # Grayscale (H, W)
            true_mask = true_masks[0].numpy()
            pred_mask = preds[0]  # shape: (14, H, W)

            # Affiche uniquement les classes présentes
            classes_to_show = np.where(labels[0].numpy() == 1)[0]

            fig, axes = plt.subplots(3, len(classes_to_show), figsize=(4 * len(classes_to_show), 10))
            if len(classes_to_show) == 1:
                axes = np.expand_dims(axes, axis=1)  # Pour gérer 1 seule classe

            for i, class_idx in enumerate(classes_to_show):
                axes[0, i].imshow(image, cmap='gray')
                axes[0, i].set_title(f"Image originale")
                axes[0, i].axis('off')

                axes[1, i].imshow(true_mask[class_idx], cmap='Reds')
                axes[1, i].set_title(f"Masque réel: {CLASS_NAMES[class_idx]}")
                axes[1, i].axis('off')

                axes[2, i].imshow(pred_mask[class_idx], cmap='Blues')
                axes[2, i].set_title(f"Masque prédit: {CLASS_NAMES[class_idx]}")
                axes[2, i].axis('off')

            plt.tight_layout()
            plt.show()

            count += 1
