# Posar-hi el codi per fer inferència -> calcular edit distance

import sys
import cv2
import torch
import os
import editdistance
from torchvision.transforms import ToTensor
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/Recognition")
from model import CharacterClassifier
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/Detection/OCR")
from ocr import segment_letters
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
import string
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05")
from params import *

# Load the trained CNN model
model = CharacterClassifier(num_classes=36)
model.load_state_dict(torch.load(model_cnn_entrenat,  map_location=torch.device('cpu')))
model.eval()

# Function to recognize text in an image
def recognize_text(image_file):
    # Obtain bounding boxes of letters in the image
    list_bbox = segment_letters(image_file, train = False)

    # Initialize an empty list to store recognized characters
    recognized_text = []

    for bbox in list_bbox:
        x1, y1, x2, y2 = bbox

        image = Image.open(image_file)
        character_crop = image.crop((x1, y1, x2, y2))

        transforms = Compose([
                    Resize((64, 64)),  # Resize the image to a specific size
                    ToTensor(),  # Convert the image to a tensor
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        
        character_crop = transforms(character_crop).unsqueeze(0)
        index_to_char = {i:k for i,k in enumerate(string.ascii_lowercase + string.digits)}

        # Pass the letter through the CNN model
        with torch.no_grad():
            outputs = model(character_crop)
            _, predicted = torch.max(outputs, 1)
            predicted_char = index_to_char[predicted.item()] # Convert class index to character

        # Append the predicted character to the recognized text
        recognized_text.append(predicted_char)

    # Join the recognized characters and return the final text
    return ''.join(recognized_text)
    
def metrics(predicted_labels, text_labels):
    accur = sum([1 if i == j else 0 for i,j in zip(predicted_labels, text_labels)])/len(predicted_labels)
    edit_dist = sum([editdistance.eval(p,t) for p,t in zip(predicted_labels, text_labels)])/len(predicted_labels)
    return edit_dist, accur

predicted_labels = []
text_labels = []
#### AMB EL NOSTRE DATASET
#for img_file in os.listdir(train_images):
#    predicted_word = recognize_text(os.path.join(train_images, img_file))
#    predicted_labels.append(predicted_word)
#    text_labels.append(img_file.split(".")[0])

#### AMB EL IIT
mapping = {str(i): char for i, char in enumerate(string.ascii_lowercase+string.digits)}
for img_file in os.listdir(train_images):
    predicted_word = recognize_text(os.path.join(train_images, img_file))
    predicted_labels.append(predicted_word)
    txt_path = os.path.join(train_labels, img_file.split(".")[0]+".txt")
    word = ""
    with open(txt_path, 'r') as file:
        for line in file:
            index = line.split(" ")[0]
            word += mapping.get(index)
    text_labels.append(word)

edit_dist, accur = metrics(predicted_labels, text_labels)
print(edit_dist)
print(accur)