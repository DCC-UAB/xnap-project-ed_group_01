from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.yaml")

results = model.train(data = '/home/alumne/ProjecteNN/xnap-project-ed_group_01/utils/yolo_utils/data.yaml', epochs = 2, task = "detect")