import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import string

class CharacterDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transforms = None):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.annotations = self._read_annotations()
        self.map_dict = {k:i for i,k in enumerate(string.ascii_lowercase + string.digits)}
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]

        image_file = annotation['image_file']
        x1, y1, x2, y2 = annotation['bbox']

        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path)
        character_crop = image.crop((x1, y1, x2, y2))

        if self.transforms != None:
            character_crop = self.transforms(character_crop)

        label = self.map_dict[annotation['character']]

        label_tensor = torch.tensor(label)

        return character_crop, label_tensor

    def _read_annotations(self):
        annotations = []
        with open(self.annotation_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_file, character_label, x1, y1, x2, y2 = line.strip().split(' ')
                bbox = list(map(float, [x1, y1, x2, y2]))
                annotation = {'bbox': bbox, 'image_file': image_file, 'character': character_label}
                annotations.append(annotation)
                #print(character_label)
                #image = Image.open(os.path.join(self.image_dir, image_file))
                #image.crop((x1, y1, x2, y2)).show()
        return annotations

#image_dir = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/data/images/train'
#annotation_file = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/data/labels/train_predict/predict_final.txt'
#dataset = CharacterDataset(annotation_file, image_dir)