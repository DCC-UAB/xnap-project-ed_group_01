from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

num_classes = 26
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.fc = torch.nn.Linear(512, num_classes) 

model.eval()

transformacions = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the character image to the expected input size of the CNN model
    transforms.ToTensor(),        # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

def segment_and_recognize(image_path, model_path):
    
    model_yolo = YOLO(model_path)
    image = Image.open(image_path)
    results = model_yolo(image)

    predictions = results[0].boxes.xyxy.numpy()

    for pred in predictions:
        x1, y1, x2, y2 = pred

        region = image.crop((x1, y1, x2, y2))

        character = transformacions(region).unsqueeze(0)

        with torch.no_grad():
            output = model(character)

        _, predicted_class = torch.max(output, 1)
        predicted_character = predicted_class.item() 

        print(predicted_character)

        #print(type(region))
        #plt.imshow(region)
        #plt.show()

image_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/test/test_img/75_Leavers_43839.jpg'
model_path = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/checkpoints/last_train7.pt'
segment_and_recognize(image_path, model_path)