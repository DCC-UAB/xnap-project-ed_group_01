import string
import os
import sys
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import editdistance
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/Recognition")
from model import CharacterClassifier
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from torchvision import transforms
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05")
from params import *

# Load the trained CNN model
model = CharacterClassifier(num_classes=36, type_model = type_model)
model.load_state_dict(torch.load(model_cnn_entrenat,  map_location=torch.device('cpu')))
model.eval()

def recognize_text(image_path, yolo_model_path):
    
    model_yolo = YOLO(yolo_model_path)
    image = Image.open(image_path)
    results = model_yolo(image)

    predictions = results[0].boxes.xyxy.numpy()
    sorted_indices = np.argsort(predictions[:, 0])
    sorted_predictions = predictions[sorted_indices]

    recognized_text = []

    for bbox in sorted_predictions:
        x1, y1, x2, y2 = bbox

        image = Image.open(image_path)
        character_crop = image.crop((x1, y1, x2, y2))

        transforms = Compose([
        Resize((64, 64)),  # Resize the image to a specific size
        ToTensor(),  # Convert the image to a tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])

        character_crop = transforms(character_crop).unsqueeze(0)
        index_to_char = {i:k for i,k in enumerate(string.ascii_lowercase + string.digits)}

        with torch.no_grad():
            outputs = model(character_crop)
            _, predicted = torch.max(outputs, 1)
            predicted_char = index_to_char[predicted.item()]

        recognized_text.append(predicted_char)

    return ''.join(recognized_text)

def metrics(predicted_labels, text_labels):
    accur = sum([1 if i == j else 0 for i,j in zip(predicted_labels, text_labels)])/len(predicted_labels)
    edit_dist = sum([editdistance.eval(p,t) for p,t in zip(predicted_labels, text_labels)])/len(predicted_labels)
    return edit_dist, accur


predicted_labels = []
text_labels = []

### AMB EL NOSTRE DATASET
#for img_file in os.listdir(train_images):
#    predicted_word = recognize_text(os.path.join(train_images, img_file), model_yolo_entrenat)
#    predicted_labels.append(predicted_word)
#    #text_labels.append(img_file.split(".")[0])
#    text_labels.append(img_file.split("_")[1])

### AMB EL IIT
mapping = {str(i): char for i, char in enumerate(string.ascii_lowercase+string.digits)}
for img_file in os.listdir(train_images):
    predicted_word = recognize_text(os.path.join(train_images, img_file), model_yolo_entrenat)
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

