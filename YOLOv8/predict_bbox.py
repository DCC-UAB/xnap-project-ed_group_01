from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

def predict_and_plot(image_path, model_path):
    # Load the model
    model = YOLO(model_path)
    image = Image.open(image_path)
    results = model(image)

    predictions = results[0].boxes.xyxy.numpy() # results[0].boxes.xyxyn.numpy()

    plt.imshow(image)
    ax = plt.gca()

    for pred in predictions:
        x, y, w, h = pred
        rect = plt.Rectangle((x, y), w, h, fill=False, color='r')
        ax.add_patch(rect)

    plt.show()

    return predictions

# Example usage
image_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/test/test_img/4.jpg'
model_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/yolov8n.pt'
predict_and_plot(image_path, model_path)