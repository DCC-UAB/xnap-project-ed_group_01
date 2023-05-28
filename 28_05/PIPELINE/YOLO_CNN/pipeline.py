import string
import os
from CNN.model import *
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from torchvision import transforms
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05")
from params import *

# Load the trained CNN model
model = CharacterClassifier(num_classes=62, type_model = type_model)
model.load_state_dict(torch.load(cnn_model_path,  map_location=torch.device('cpu')))
model.eval()

def recognize_text(image_path, yolo_model_path):
    
    model_yolo = YOLO(yolo_model_path)
    image = Image.open(image_path)
    results = model_yolo(image)

    predictions = results[0].boxes.xyxy.numpy()
    sorted_indices = np.argsort(predictions[:, 0])
    sorted_predictions = predictions[sorted_indices]

    recognized_text = []

    for bbox in sorted_predictions:
        x1, y1, x2, y2 = bbox

        image = Image.open(image_file)
        character_crop = image.crop((x1, y1, x2, y2))

        transforms = Compose([
        Resize((64, 64)),  # Resize the image to a specific size
        ToTensor(),  # Convert the image to a tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])

        character_crop = transforms(character_crop).unsqueeze(0)
        index_to_char = {i:k for i,k in enumerate(string.ascii_lowercase + string.ascii_uppercase + string.digits)}

        with torch.no_grad():
            outputs = model(character_crop)
            _, predicted = torch.max(outputs, 1)
            predicted_char = index_to_char[predicted.item()]

        recognized_text.append(predicted_char)

    return ''.join(recognized_text)

def metrics(predicted_labels, text_labels):
    accur = sum([1 for i,j in zip(predicted_labels, text_labels) if i == j else 0])/len(predicted_labels)
    edit_dist = sum([editdistance.eval(p,t) for p,t in zip(predicted_labels, text_labels)])/len(predicted_labels)
    return edit_dist, accur

predicted_labels = []
text_labels = []
for img_file in os.listdir(test_images):
    predicted_word = recognize_text(os.path.join(test_images, image_file))
    predicted_labels.append(predicted_word)
    text_labels.append(img_file.split(".")[0])

edit_dit, accur = metrics(predicted_labels, text_labels)