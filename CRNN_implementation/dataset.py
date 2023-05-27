import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TextRecognitionDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform
        self.image_paths, self.labels = self._load_labels()

    def _load_labels(self):
        image_paths = []
        labels = []

        with open(self.label_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_file = line.strip()
                image_path = os.path.join(self.image_dir, image_file)
                image_paths.append(image_path)
                labels.append(image_file.split('.')[0])  # Remove the file extension

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def _encode_label(self, label):
        encoded_label = [ord(c) - ord('a') + 1 for c in label]  # Encode each character as a number (a=1, b=2, ...)

        # Apply padding if the label is shorter than the maximum length
        max_label_length = max(len(encoded) for encoded in self.labels)
        encoded_label += [0] * (max_label_length - len(encoded_label))

        encoded_label = torch.tensor(encoded_label, dtype=torch.long)
        return encoded_label

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert('L')  # Convert to grayscale

        if self.transform is not None:
            image = self.transform(image)

        label_tensor = self._encode_label(label)
        return image, label_tensor

# Set the path to the directory containing the images
image_dir = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/CRNN_implementation/dataset/img'

# Set the path to the text file containing the labels
label_file = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/CRNN_implementation/dataset/labels.txt'

# Define the transformations to apply to the images


# Create an instance of the TextRecognitionDataset
#dataset = TextRecognitionDataset(image_dir, label_file, transform)
#aux = dataset[0]
#print("a")