from ultralytics import YOLO

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
model = YOLO('yolov8.yaml')

results = model.train(data = "/home/alumne/xnap-project-ed_group_01/28_05/PIPELINE/YOLO/data_recog.yaml", epochs = 5, task = "detect")