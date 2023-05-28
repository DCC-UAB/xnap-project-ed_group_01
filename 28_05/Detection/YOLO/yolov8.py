from ultralytics import YOLO
import sys
sys.path.insert(0, "/home/alumne/ProjecteNN/xnap-project-ed_group_01/28_05")
from params import *

# Load model
model = YOLO("yolov8n.yaml")

results = model.train(data = data_yaml, epochs = 5, task = "detect")