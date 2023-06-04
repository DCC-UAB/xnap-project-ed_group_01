from ultralytics import YOLO
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from params import *
import yaml

info = {
    "path": path_yolo_pipeline,
    "train": train_images_yolo_pipeline,
    "val": test_images_yolo_pipeline,
    "nc": 36,
    "names": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
}
with open(data_yaml_pipeline, 'w') as outfile:
    yaml.dump(info, outfile, default_flow_style=True)

# Load model
model = YOLO('yolov8.yaml')

results = model.train(data = data_yaml_pipeline, epochs = 5, task = "detect")