import torch
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from ultralytics import YOLO
import cv2

#Model
model = YOLO('yolov8n.pt')

# Images
img = cv2.imread('./utils/yolo_utils/ouput.png')

# Inference
#results = model(img)
results = model.predict(img)
#results = model.predict("./utils/yolo_utils/6_1.png")
result = results[0]
#print(results)
print(len(result.boxes))

# Results
#results.print()  
#results.show()  # or .show()

#results = results.xyxy[0]  # img1 predictions (tensor)
#boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)