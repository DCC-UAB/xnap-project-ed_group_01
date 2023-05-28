from ultralytics import YOLO
import sys
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05")
from params import *

# Load model
model = YOLO("yolov8n.yaml")

results = model.train(data = data_yaml, epochs = 3, task = "detect")