import sys
sys.path.insert(0, "/Users/xavid/Documents/GitHub/xnap-project-ed_group_01/28_05")
from params import *
from PIL import Image, ImageDraw
import os

def convert(a):
    new_a = a.copy()
    new_a[0] = a[0] - a[2]/2
    new_a[1] = a[1] - a[3]/2
    new_a[2] = a[0] + a[2]/2
    new_a[3] = a[1] + a[3]/2
    return new_a

def get_iou(a, b, epsilon=1e-5):

    a = convert(a)
    b = convert(b)

    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


def calculate_metrics(gt_bboxes, pred_bboxes, th):
    tp = 0
    fn = 0
    fp = 0

    d = {}
    for i in range(len(pred_bboxes)):
        d[i]= "fp"

    for j in range(len(gt_bboxes)):
        values = []
        indices = []
        for i in range(len(pred_bboxes)):
            iou = get_iou(gt_bboxes[j][1:], pred_bboxes[i])
            if iou>th:
                values.append(iou)
                indices.append(i)
        if len(values) >0:
            index = values.index(max(values))
            for ind in indices:
                if ind == indices[index]:
                    d[ind] = "tp"
        else:
            fn += 1
    
    tp = sum([1 for v in d.values() if v == "tp"])
    
    fp = sum([1 for v in d.values() if v == "fp"])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return tp, fp, fn, tp+fp+fn, precision, recall




def read_txt_file(file_path):
    result = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            numbers = [float(num) for num in line.strip().split()]
            result.append(numbers)
    return result

# Aquí només ho estem fent per una imatge, la idea seria tenir-hi el directori amb predictions i labels correctes
# i que iteri per tots els elmeents dels directoris de 1 en 1 i calculi metriquis i dongui una preciosn, recall general per TOT
file_path_prediction = ocr_predictions 
file_path_labels = test_labels

all_precision = []
all_recall = []

import numpy as np

for th in np.arange(0.4):
    precision = []
    recall = []
    for i,filename in enumerate(os.listdir(file_path_labels)):
        gt_file = os.path.join(file_path_labels, filename)
        pred_file = os.path.join(file_path_prediction, filename)

        gt_bboxes = read_txt_file(gt_file)
        pred_bboxes = read_txt_file(pred_file)
        
        tp_, fp_, fn_, total, precision_, recall_ =  calculate_metrics(gt_bboxes, pred_bboxes, th)

        precision.append(precision_)
        recall.append(recall_)
        if i%100 == 0:
            print(i)
    all_precision.append(sum(precision)/len(precision))
    all_recall.append(sum(recall)/len(recall))
    print("th")


print(all_precision)
print(all_recall)