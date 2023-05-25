from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.yaml")

results = model.train(data = '/home/alumne/ProjecteNN/xnap-project-ed_group_01/YOLO_recognition/data.yaml', epochs = 10, task = "detect")