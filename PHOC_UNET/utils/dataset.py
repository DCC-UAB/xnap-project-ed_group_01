from torch.utils.data import Dataset
from torchvision.io import read_image
import torch

from .build_phoc import phoc

class dataset(Dataset):
    def __init__(self, annotations_file, img_dir, train, transform = None):

        with open(annotations_file, "r") as file:
            self.paths = file.readlines()
        self.img_dir = img_dir
        self.transform = transform
        if train:
            self.paths = self.paths[:5000]
        else:
            self.paths = self.paths[-1000:]
    def __len__(self):

        return len(self.paths)
    
    def __getitem__(self, idx):
        if "xavid" in self.img_dir or "abriil" in self.img_dir:
            path = self.img_dir + self.paths[idx].split("\n")[0].split(" ")[0][2:]
        else:
            path = self.img_dir + self.paths[idx].split("\n")[0]
        img = read_image(path)#[0,:,:]
        #img = img.reshape([1,img.shape[0],img.shape[1]])
        word = self.paths[idx].split("_")[1]
        target = phoc(word)
        img = img.to(torch.float32)
        if self.transform != None:
            img = self.transform(img)
        img /= 255
        
        target = target.reshape(target.shape[1])

        return img, target, word