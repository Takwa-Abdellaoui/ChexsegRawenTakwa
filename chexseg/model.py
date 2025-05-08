import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


class ChexNetEncoder(nn.Module):
    def __init__(self, num_classes):
        super(ChexNetEncoder, self).__init__()

        # Charger et stocker le modèle de base
        self.densenet = models.densenet121(pretrained=False)

        # Charger les poids
        checkpoint = torch.load("model.pth.tar", map_location="cpu")
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module.densenet121."):
                new_key = k[len("module.densenet121."):]
                new_state_dict[new_key] = v

        self.densenet.load_state_dict(new_state_dict, strict=False)

        # Modifier le classifieur pour 14 sorties au lieu de 1000
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()
        )

        # Conserver les features pour l’encodeur U-Net
        self.features = self.densenet.features

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        probabilities = self.densenet.classifier(out)
        return features, probabilities


    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        probabilities = self.densenet.classifier(out)
        return features, probabilities


  
    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        probabilities = self.densenet.classifier(out)
        return features, probabilities


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class ChexNetUNet(nn.Module):
    def __init__(self, num_classes=14, num_masks=14):
        super(ChexNetUNet, self).__init__()
        self.encoder = ChexNetEncoder(num_classes)

        self.decoder = nn.Sequential(
            DecoderBlock(1024, 512),
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            nn.Conv2d(64, num_masks, kernel_size=1)
        )

    def forward(self, x):
        features, class_probs = self.encoder(x)
        masks = self.decoder(features)
        masks = torch.sigmoid(masks)
        return masks, class_probs
