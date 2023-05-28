# Posar-hi el codi per fer inferència -> calcular edit distance

import cv2
import torch
from torchvision.transforms import ToTensor
from model import CharacterClassifier
from segment_words import segment_letters
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
import string
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05")
from params import *

# Load the trained CNN model
model = CharacterClassifier(num_classes=62, type_model = type_model)
model.load_state_dict(torch.load(mode_cnn_entrenat,  map_location=torch.device('cpu')))
model.eval()

# Function to recognize text in an image
def recognize_text(image_file):
    # Obtain bounding boxes of letters in the image
    list_bbox = segment_letters(image_file, train = False)

    # Initialize an empty list to store recognized characters
    recognized_text = []

    for bbox in list_bbox:
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

        # Pass the letter through the CNN model
        with torch.no_grad():
            outputs = model(character_crop)
            _, predicted = torch.max(outputs, 1)
            predicted_char = index_to_char[predicted.item()] # Convert class index to character

        # Append the predicted character to the recognized text
        recognized_text.append(predicted_char)

    # Join the recognized characters and return the final text
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
