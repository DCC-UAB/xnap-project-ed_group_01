import string
import os
import sys
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import editdistance
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'\\Recognition')
from model import CharacterClassifier
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from params import *

# Load the trained CNN model
model = CharacterClassifier(num_classes=36)
model.load_state_dict(torch.load(model_cnn_entrenat,  map_location=torch.device('cpu')))
model.eval()

def recognize_text(image_path, yolo_model_path):
    
    model_yolo = YOLO(yolo_model_path)
    image = Image.open(image_path)
    results = model_yolo(image)

    predictions = results[0].boxes.xyxy.cpu().data.numpy()
    sorted_indices = np.argsort(predictions[:, 0])
    list_bbox = predictions[sorted_indices]

    recognized_text = []

    for bbox in list_bbox:
        x1, y1, x2, y2 = bbox

        image = Image.open(image_path)
        character_crop = image.crop((x1, y1, x2, y2))

        transforms = Compose([
        Resize((64, 64)),  # Resize the image to a specific size
        ToTensor(),  # Convert the image to a tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])

        try:
            character_crop = transforms(character_crop).unsqueeze(0)
        except:
            return None
        index_to_char = {i:k for i,k in enumerate(string.ascii_lowercase + string.digits)}

        with torch.no_grad():
            outputs = model(character_crop)
            _, predicted = torch.max(outputs, 1)
            predicted_char = index_to_char[predicted.item()]

        recognized_text.append(predicted_char)

    return ''.join(recognized_text)

def metrics(predicted_labels, text_labels):
    accur = sum([1 if i == j else 0 for i,j in zip(predicted_labels, text_labels)])/len(predicted_labels)
    accur_2 = sum([1 if editdistance.eval(i,j) < 2 else 0 for i,j in zip(predicted_labels, text_labels)])/len(predicted_labels)
    edit_dist = sum([editdistance.eval(p,t) for p,t in zip(predicted_labels, text_labels)])/len(predicted_labels)
    return edit_dist, accur, accur_2


predicted_labels = []
text_labels = []
images_paths = []
file_list = glob.glob(test_images + '/*')
file_count = len(file_list)

if dataset == "ours":
    for i,img_file in enumerate(os.listdir(test_images)):
        predicted_word = recognize_text(os.path.join(test_images, img_file), model_yolo_entrenat)
        predicted_labels.append(predicted_word)
        text_labels.append(img_file.split(".")[0])
        #text_labels.append(img_file.split("_")[1])
        images_paths.append(os.path.join(test_images, img_file))
        #print(f"{i}/{file_count}")

elif dataset == "iiit":
    mapping = {str(i): char for i, char in enumerate(string.ascii_lowercase+string.digits)}
    for i,img_file in enumerate(os.listdir(test_images)):
        predicted_word = recognize_text(os.path.join(test_images, img_file), model_yolo_entrenat)
        if predicted_word == None:
            continue
        predicted_labels.append(predicted_word)
        txt_path = os.path.join(test_labels, img_file.split(".")[0]+".txt")
        word = ""
        with open(txt_path, 'r') as file:
            for line in file:
                index = line.split(" ")[0]
                word += mapping.get(index)
        text_labels.append(word)
        #print(f"{i}/{file_count}")

edit_dist, accur, accur2 = metrics(predicted_labels, text_labels)
print(f"Edit distance: {edit_dist}")
print(f"Accuracy: {accur}")
print(f"Accuracy with edit_dist < 2: {accur2}")
with open('predicted_labels.txt', 'w') as file:
    # Write each item in the list to a new line in the file
    for item in predicted_labels:
        file.write(item + '\n')
with open('images_paths.txt', 'w') as file:
    # Write each item in the list to a new line in the file
    for item in images_paths:
        file.write(item + '\n')

# Get all unique letters from both ground truth and predicted words
letters = list(set(''.join(text_labels + predicted_labels)))

# Create a confusion matrix of zeros
confusion_mat = np.zeros((len(letters), len(letters)), dtype=int)

# Iterate over each word pair
for gt_word, pred_word in zip(text_labels, predicted_labels):
    # Iterate over each letter pair
    for gt_letter, pred_letter in zip(gt_word, pred_word):
        # Increment the corresponding cell in the confusion matrix
        gt_index = letters.index(gt_letter)
        pred_index = letters.index(pred_letter)
        confusion_mat[gt_index][pred_index] += 1

confusion_df = pd.DataFrame(confusion_mat, index=letters, columns=letters)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_df, annot=False, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.title('Confusion Matrix YOLO + CNN')
plt.show()