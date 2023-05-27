import cv2
import os
import pytesseract
from PIL import Image, ImageDraw
import string


def segment_letters(image_path):
    image = cv2.imread(image_path)
    img_original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        thresh_inv = 255 - thresh
        contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        letter_bboxes.append((x, y, x+w, y+h))

    letter_bboxes.sort(key=lambda bbox: bbox[0])

    letters = os.path.splitext(image_path)[0].split("_")[1]
    annotations = []
    string_aux = ""

    for letter,bbox in zip(letters, letter_bboxes):
        x1, y1, x2, y2 = bbox
        annotation = f"{image_path} {letter} {x1} {y1} {x2} {y2}"
        annotations.append(annotation)
        string_aux += letter

    with open("C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/OCR_cnn/annotation.txt", 'a') as f:
        f.write('\n'.join(annotations))
        f.write('\n')

directory  = 'C:/Users/adars/OneDrive/Escritorio/ProjecteNN/mnt/ramdisk/max/90kDICT32px/1/1'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    segment_letters(f)

    #print(bounding_boxes)