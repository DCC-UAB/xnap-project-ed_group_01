from ultralytics import YOLO
from PIL import Image, ImageDraw
import sys
import os
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05")
from params import *
import editdistance

def calculate_metrics(image_path, model_path):
    model = YOLO(model_path)

    predicted_labels = []
    text_labels = []

    for img_file in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, img_file))
        gt_label = img_file.split(".")[0] 

        results = model.predict(img)

        predictions = results[0].boxes.xyxy.numpy()
        sorted_indices = np.argsort(predictions[:, 0])
        sorted_predictions = results[0].boxes.cls[sorted_indices]

        word_prediction = ""
        for c in sorted_predictions:
            word_prediction += model.names[c]

        predicted_labels.append(word_prediction)
        text_labels.append(gt_label)

    edit_dist, accur =  metrics(predicted_labels, text_labels)

    return accur, edit_dist

def metrics(predicted_labels, text_labels):
    accur = sum([1 for i,j in zip(predicted_labels, text_labels) if i == j else 0])/len(predicted_labels)
    edit_dist = sum([editdistance.eval(p,t) for p,t in zip(predicted_labels, text_labels)])/len(predicted_labels)
    return edit_dist, accur
