from torch.utils.data import Dataset
from torchvision.io import read_image
import torch

from .build_phoc import phoc

class dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None):

        with open(annotations_file, "r") as file:
            self.paths = file.readlines()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):

        return len(self.paths)
    
    def __getitem__(self, idx):
        if "xavid" in self.img_dir:
            path = self.img_dir + self.paths[idx].split("\n")[0].split(" ")[0][2:]
        else:
            path = self.img_dir + self.paths[idx].split("\n")[0]
        img = read_image(path)
        word = self.paths[idx].split("_")[1]
        target = phoc(word)
        img = img.to(torch.float32)
        if self.transform != None:
            img = self.transform(img)
        img /= 255
        
        target = target.reshape([604])

        return img, target, word