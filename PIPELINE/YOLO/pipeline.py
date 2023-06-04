from ultralytics import YOLO
from PIL import Image
import sys
import os
from ultralytics import YOLO
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from params import *
import editdistance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import string

def recognize_text(image_path, yolo_model_path, store_files, dataset = "normal"):
    model = YOLO(yolo_model_path)

    predicted_labels = []
    text_labels = []
    images_paths = []
    file_list = glob.glob(image_path + '/*')
    file_count = len(file_list)

    if dataset == "normal":
        for i,img_file in enumerate(os.listdir(image_path)):
            img = Image.open(os.path.join(image_path, img_file))
            gt_label = img_file.split(".")[0] 

            results = model.predict(img)

            predictions = results[0].boxes.xyxy.cpu().data.numpy()
            sorted_indices = np.argsort(predictions[:, 0])
            sorted_predictions = results[0].boxes.cls[sorted_indices]

            word_prediction = ""
            for c in sorted_predictions:
                word_prediction += model.names[c.item()]

            predicted_labels.append(word_prediction)
            text_labels.append(gt_label)
            images_paths.append(os.path.join(test_images, img_file))
            print(f"{i}/{file_count}")
    elif dataset == "iit":
        mapping = {str(i): char for i, char in enumerate(string.ascii_lowercase+string.digits)}
        for i,img_file in enumerate(os.listdir(image_path)):
            img = Image.open(os.path.join(image_path, img_file))

            results = model.predict(img)

            predictions = results[0].boxes.xyxy.cpu().data.numpy()
            sorted_indices = np.argsort(predictions[:, 0])
            sorted_predictions = results[0].boxes.cls[sorted_indices]

            word_prediction = ""
            for c in sorted_predictions:
                word_prediction += model.names[c.item()]

            predicted_labels.append(word_prediction)

            txt_path = os.path.join(test_labels, img_file.split(".")[0]+".txt")
            word = ""
            with open(txt_path, 'r') as file:
                for line in file:
                    index = line.split(" ")[0]
                    word += mapping.get(index)
            text_labels.append(word)
            print(f"{i}/{file_count}")

    edit_dist, accur, accur_2 =  metrics(predicted_labels, text_labels)

    with open(store_files +'/predicted_labels_dif.txt', 'w') as file:
    # Write each item in the list to a new line in the file
        for item in predicted_labels:
            file.write(item + '\n')
    with open(store_files+'/images_paths_dif.txt', 'w') as file:
    # Write each item in the list to a new line in the file
        for item in images_paths:
            file.write(item + '\n')

    return edit_dist, accur, accur_2, predicted_labels, text_labels

def metrics(predicted_labels, text_labels):
    accur = sum([1 if i == j else 0 for i,j in zip(predicted_labels, text_labels)])/len(predicted_labels)
    accur_2 = sum([1 if editdistance.eval(i,j) < 1 else 0 for i,j in zip(predicted_labels, text_labels)])/len(predicted_labels)
    edit_dist = sum([editdistance.eval(p,t) for p,t in zip(predicted_labels, text_labels)])/len(predicted_labels)
    return edit_dist, accur, accur_2

#predicted_labels = []
#text_labels = []
#for img_file in os.listdir(train_images):
#    predicted_word = recognize_text(os.path.join(train_images, img_file), yolo_entrenat_recog_detect)
#    predicted_labels.append(predicted_word)
#    text_labels.append(img_file.split(".")[0])

edit_dist, accur, accur2, predicted_labels, text_labels = recognize_text(test_images, yolo_entrenat_recog_detect, yolo_pipeline_store_files, "iit")
print(f"Edit distance: {edit_dist}")
print(f"Accuracy: {accur}")

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
plt.title('Confusion Matrix YOLO')
plt.show()