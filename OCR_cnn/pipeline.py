# Posar-hi el codi per fer inferència -> calcular edit distance

import cv2
import torch
from torchvision.transforms import ToTensor
from model import CharacterClassifier
from segment_words import segment_letters
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
import string

# Load the trained CNN model
model = CharacterClassifier(num_classes=62)
model.load_state_dict(torch.load('C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/OCR_cnn/saved_model/model.pt',  map_location=torch.device('cpu')))
model.eval()

# Function to recognize text in an image
def recognize_text(image):
    # Obtain bounding boxes of letters in the image
    segment_letters(image, "test")

    with open('C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/OCR_cnn/annotation_uwu.txt', 'r') as file:
        lines = file.readlines()

    # Initialize an empty list to store recognized characters
    recognized_text = []

    for line in lines:
        # Extract relevant information from the line
        image_file, x1, y1, x2, y2 = line.strip().split(' ')
        x1, y1, x2, y2 = list(map(float, [x1, y1, x2, y2]))

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


# Example usage
image_path = directory = "C:/Users/adars/OneDrive/Escritorio/ProjecteNN/mnt/ramdisk/max/90kDICT32px/1/2/26_nudes_52518.jpg"

recognized_text = recognize_text(image_path)
print("Recognized Text:", recognized_text)


