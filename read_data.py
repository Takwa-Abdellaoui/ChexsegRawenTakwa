# encoding: utf-8

"""
Read images and corresponding labels.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_names = []
        self.labels = []

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                self.image_names.append(image_name)
                self.labels.append(label)

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open('C:\\Users\\Admin\\Downloads\\ChestX-ray14\\images\\images\\00006596_004.png').convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)