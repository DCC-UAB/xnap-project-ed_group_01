import os
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import string
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import *

class CharacterDataset(Dataset):
    def __init__(self, annotation_dir, image_dir, transforms = None):
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.annotations, self.file_paths = self._read_annotations()
        self.map_dict = {k:i for i,k in enumerate(string.ascii_lowercase + string.ascii_uppercase + string.digits)}
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def convert(self, a, w, h):
        new_a = a.copy()
        new_a[0] = (a[0] - a[2]/2) * w
        new_a[1] = (a[1] - a[3]/2) * h
        new_a[2] = (a[0] + a[2]/2) * w
        new_a[3] = (a[1] + a[3]/2) * h 
        return new_a

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image_file = self.file_paths[index]

        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path)
        #draw = ImageDraw.Draw(image)
        w, h = image.size
        x1, y1, x2, y2 = self.convert(annotation['bbox'], w, h)
        #draw.rectangle((x1, y1, x2, y2), None, "#f00")
        ###image.show()
        character_crop = image.crop((x1, y1, x2, y2))

        if self.transforms != None:
            character_crop = self.transforms(character_crop)

        label = int(annotation['character'])

        label_tensor = torch.tensor(label)

        return character_crop, label_tensor

    def _read_annotations(self):
        annotations = []
        file_path = []
        for annotation_file in os.listdir(self.annotation_dir):
            with open(os.path.join(self.annotation_dir, annotation_file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    character_label, x1, y1, x2, y2 = line.strip().split(' ')
                    bbox = list(map(float, [x1, y1, x2, y2]))
                    annotation = {'bbox': bbox, 'character': character_label}
                    annotations.append(annotation)
                    file_path.append(annotation_file.split(".")[0] + ".jpg")
        return annotations, file_path