from ultralytics import YOLO
from PIL import Image, ImageDraw

def predict_and_plot(image_path, model_path):
    # Load the model
    model = YOLO(model_path)
    #model = YOLO('yolov8n.yaml').load(model_path)
    #model.conf = 0.1
    image = Image.open(image_path)
    #results = model.predict(image, imgsz=1000, augment = True, conf=0.03, iou = 0.01)
    results = model.predict(image)

    predictions = results[0].boxes.xyxy.numpy() # results[0].boxes.xyxyn.numpy()
    predictions_norm = results[0].boxes.xywhn.numpy()
    
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        x1, y1, x2, y2 = pred
        draw.rectangle((x1,y1,x2,y2), None, "#f00")

    image.show()

    return predictions_norm

# Example usage
image_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLO_recognition/test/test_img/am3j9kz62a.jpg'
model_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLO_recognition/checkpoints/last.pt'
bb_predictions = predict_and_plot(image_path, model_path)
print(bb_predictions)