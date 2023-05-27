import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from phoc import *

class PHOCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])

        
        self.file_list = sorted(os.listdir(root_dir))
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root_dir, filename)
        
        # Load image
        image = Image.open(image_path)
        image = self.transform(image)
        
        # Get label (word in the image)
        label = os.path.splitext(filename)[0].split("_")[1]
        label = label.lower()
        
        # Convert label to PHOC representation
        phoc = build_phoc([label], 'abcdefghijklmnopqrstuvwxyz0123456789', [1])[0]
        
        return image, phoc