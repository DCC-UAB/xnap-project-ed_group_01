import torch
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from ultralytics import YOLO
import cv2

model_path = "/home/alumne/ProjecteNN/xnap-project-ed_group_01/YOLOv8/yolov8n.pt"
img_path = "/home/alumne/data/images/test/9.jpg"
model = YOLO(model_path)
img = cv2.imread(img_path)

results = model.predict(img)
result = results[0]
print(result.boxes)

def predict(model_path, img_path):

    model = YOLO(model_path)
    img = cv2.imread(img_path)

    results = model.predict(img)
    result = results[0]
    print(len(result.boxes))

# Results
#results.print()  
#results.show()  # or .show()

#results = results.xyxy[0]  # img1 predictions (tensor)
#boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)