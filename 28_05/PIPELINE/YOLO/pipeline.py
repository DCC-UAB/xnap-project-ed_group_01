from ultralytics import YOLO
from PIL import Image, ImageDraw
import sys
import os
from ultralytics import YOLO
import numpy as np
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05")
from params import *
import editdistance
import string

def recognize_text(image_path, yolo_model_path, dataset = "normal"):
    model = YOLO(yolo_model_path)

    predicted_labels = []
    text_labels = []

    if dataset == "normal":
        for img_file in os.listdir(image_path):
            img = Image.open(os.path.join(image_path, img_file))
            gt_label = img_file.split(".")[0] 

            results = model.predict(img)

            predictions = results[0].boxes.xyxy.numpy()
            sorted_indices = np.argsort(predictions[:, 0])
            sorted_predictions = results[0].boxes.cls[sorted_indices]

            word_prediction = ""
            for c in sorted_predictions:
                word_prediction += model.names[c.item()]

            predicted_labels.append(word_prediction)
            text_labels.append(gt_label)
    elif dataset == "iit":
        mapping = {str(i): char for i, char in enumerate(string.ascii_lowercase+string.digits)}
        for img_file in os.listdir(train_images):
            img = Image.open(os.path.join(image_path, img_file))

            results = model.predict(img)

            predictions = results[0].boxes.xyxy.numpy()
            sorted_indices = np.argsort(predictions[:, 0])
            sorted_predictions = results[0].boxes.cls[sorted_indices]

            word_prediction = ""
            for c in sorted_predictions:
                word_prediction += model.names[c.item()]

            predicted_labels.append(word_prediction)

            txt_path = os.path.join(train_labels, img_file.split(".")[0]+".txt")
            word = ""
            with open(txt_path, 'r') as file:
                for line in file:
                    index = line.split(" ")[0]
                    word += mapping.get(index)
            text_labels.append(word)

    edit_dist, accur =  metrics(predicted_labels, text_labels)

    return accur, edit_dist

def metrics(predicted_labels, text_labels):
    accur = sum([1 if i == j else 0 for i,j in zip(predicted_labels, text_labels)])/len(predicted_labels)
    edit_dist = sum([editdistance.eval(p,t) for p,t in zip(predicted_labels, text_labels)])/len(predicted_labels)
    return edit_dist, accur

#predicted_labels = []
#text_labels = []
#for img_file in os.listdir(train_images):
#    predicted_word = recognize_text(os.path.join(train_images, img_file), yolo_entrenat_recog_detect)
#    predicted_labels.append(predicted_word)
#    text_labels.append(img_file.split(".")[0])

edit_dist, accur = recognize_text(train_images, yolo_entrenat_recog_detect, "iit")
print(edit_dist)
print(accur)

