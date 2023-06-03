from ultralytics import YOLO

# Load model
model = YOLO('yolov8.yaml')

results = model.train(data = "/home/alumne/xnap-project-ed_group_01/28_05/PIPELINE/YOLO/data_recog.yaml", epochs = 5, task = "detect")