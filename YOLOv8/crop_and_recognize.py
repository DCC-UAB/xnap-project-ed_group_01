from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import numpy as np
import string
import os
from CNN.model import *
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from torchvision import transforms


def segment_and_recognize(image_path, yolo_model_path, cnn_model_path):
    
    model_yolo = YOLO(yolo_model_path)
    image = Image.open(image_path)
    results = model_yolo(image)

    predictions = results[0].boxes.xyxy.numpy()
    sorted_indices = np.argsort(predictions[:, 0])
    sorted_predictions = predictions[sorted_indices]

    model_cnn = CharacterClassifier(num_classes=36)
    model_cnn.load_state_dict(torch.load(cnn_model_path))

    transforms = Compose([
    Resize((64, 64)),  # Resize the image to a specific size
    ToTensor(),  # Convert the image to a tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    model_cnn.eval()
    output_string = ""
    map_dict = {i:k for i,k in enumerate(string.ascii_lowercase + string.digits)}
    for pred in sorted_predictions:
        x1, y1, x2, y2 = pred
        character_crop = image.crop((x1, y1, x2, y2))
        character_crop = transforms(character_crop)
        with torch.no_grad():
            input_batch = character_crop.unsqueeze(0)
            output = model_cnn(input_batch)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        class_index = predicted_class.item()
        output_string += map_dict[class_index]

    print(output_string)
    image.show()
    return output_string


def generate_annotations(image_folder, model_path, txt_file):

    model_yolo = YOLO(model_path)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with open(txt_file, "w") as f:
        for i,image_file in enumerate(image_files):

            image_path = os.path.join(image_folder, image_file)

            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            results = model_yolo(image)

            predictions = results[0].boxes.xyxy.numpy()
            sorted_indices = np.argsort(predictions[:, 0])
            sorted_predictions = predictions[sorted_indices]

            for idx,pred in enumerate(sorted_predictions):
                x1, y1, x2, y2 = pred
                draw.rectangle((x1, y1, x2, y2), None, "#f00")
                line = f"{image_file} {image_file[idx]} {x1} {y1} {x2} {y2}\n"
                f.write(line)
            image.show()

image_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/img'
yolo_model_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/checkpoints/last_train11.pt'
save_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/img/predict_final.txt'
generate_annotations(image_path, yolo_model_path, save_path)
#img_p = '/home/alumne/data/images/test/0akdxsw0.jpg'
#cnn_model_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/CNN/saved_model/model.pt'
#segment_and_recognize(img_p, yolo_model_path, cnn_model_path)
