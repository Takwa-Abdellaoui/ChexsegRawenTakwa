import torch
import argparse
import os
from chexseg.chexnet_model import DenseNet121
from read_data import ChestXrayDataSet
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./model.pth.tar')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()

    # Vérification du GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations des images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Chargement des données de test
    test_dataset = ChestXrayDataSet(
        data_dir=args.data_dir,
        image_list_file=os.path.join(args.data_dir, 'test_list.txt'),
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Chargement du modèle
    model = DenseNet121(14).to(device)
    
    # Chargement des poids avec gestion de DataParallel
    state_dict = torch.load(args.model_path, map_location=device)['state_dict']
    
    # Nettoyage des clés si le modèle a été sauvegardé avec DataParallel
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()

    # Test du modèle
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # Affichage des informations de batch
            print(f"Batch {batch_idx + 1}/{len(test_loader)}")
            print(f"Input shape: {inputs.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Labels shape: {labels.shape}")
            print("-" * 50)

if __name__ == '__main__':
    main()