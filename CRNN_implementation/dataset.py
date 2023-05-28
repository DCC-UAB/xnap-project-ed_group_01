import albumentations
import torch
import os
import numpy as np

from PIL import Image
from PIL import ImageFile
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import string

class Dataset(Dataset):

    def __init__(self, img_dir, resize = (32,128)):
        path_list = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir)
        self.resize = resize
        self.max_label_len = 0

        self.img = []
        self.orig_txt = []
        self.label_length = []
        self.input_length = []
        self.txt = []

        for path in path_list:
            
            img = cv2.cvtColor(cv2.imread(os.path.join(abspath, path)), cv2.COLOR_BGR2GRAY) 

            img = cv2.resize(img, (32, 128))
            img = np.expand_dims(img , axis = 2)
            img = np.transpose(img,(2, 1, 0))
            img = img/255.

            label = os.path.basename(os.path.join(abspath, path)).split('.')[0].lower().strip()
            self.img.append(img)
            self.orig_txt.append(label)
            self.input_length.append(31)
            self.label_length.append(len(label))
            self.txt.append(self.encode_to_labels(label))

            if len(label) > self.max_label_len:
                self.max_label_len = len(label)

        # AMB VARIABLE LENGTH CAL FER PADDING THE SELF.TXT!!!!!!!!        

 
    def encode_to_labels(self, txt):
        # encoding each output word into digits
        char_list = string.ascii_letters+string.digits
        dig_lst = []
        for index, char in enumerate(txt):
            try:
                dig_lst.append(char_list.index(char))
            except:
                print(char)
        
        return dig_lst

    def __len__(self):
        return len(self.img)
    
    
    def __getitem__(self, idx):
        img_vector = torch.tensor(self.img[idx], dtype = torch.float32)
        orig_txt = self.orig_txt[idx]
        label_length_vect = torch.tensor(self.label_length[idx], dtype = torch.int32)
        input_length_vector = torch.tensor(self.input_length[idx], dtype = torch.int32)
        txt = torch.tensor(self.txt[idx], dtype = torch.long)

        return img_vector, txt, orig_txt, label_length_vect, input_length_vector

#aux = Dataset("C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/CRNN_implementation/dataset/img2/test")
#aux[0]
#train_loader = DataLoader(
#    aux, batch_size=16, shuffle=True
#)
#print("a")