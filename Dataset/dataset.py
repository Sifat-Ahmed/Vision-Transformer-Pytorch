import os
import cv2
import torch
from torch.utils.data import Dataset

class SmokeDataset(Dataset):
    def __init__(self, images, labels, resize = False, transform=None, return_image_path = False):
        self._images = images
        self._labels = labels
        self._resize = resize
        self.transform = transform
        self._return_image_path = return_image_path
        
        assert len(self._images) == len(self._labels), "Number of images and labels are not same."
        
        
    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image_filepath = self._images[idx]
        label = self._labels[idx]
        
        image = cv2.imread(image_filepath)
        if self._resize:
            image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        if self._return_image_path:
            return image, image_filepath, torch.tensor(label, dtype= torch.float)
        
        return image, torch.tensor(label, dtype= torch.float)