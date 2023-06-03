from ultralytics import YOLO
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from params import *
import yaml

info = {
    "path": path_yolo,
    "train": train_images,
    "val": test_images,
    "nc": 1,
    "names": ['character']
}
with open(data_yaml, 'w') as outfile:
    yaml.dump(info, outfile, default_flow_style=True)

# Load model
model = YOLO("yolov8n.yaml")

results = model.train(data = data_yaml, epochs = 50, task = "detect")