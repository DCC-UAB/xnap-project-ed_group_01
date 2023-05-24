from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.yaml")

results = model.train(data = '/home/alumne/ProjecteNN/xnap-project-ed_group_01/YOLOv8/data.yaml', epochs = 5, task = "detect")