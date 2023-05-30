import cv2
import os
#import pytesseract
from PIL import Image, ImageDraw
import string
import glob
import sys
sys.path.insert(0, "/home/alumne/ProjecteNN/xnap-project-ed_group_01/28_05")
from params import *

def convert_bbox_to_yolo(bbox, image_width, image_height):
    x, y, width, height = bbox

    # Calculate the bounding box center coordinates
    center_x = x + width / 2
    center_y = y + height / 2

    # Normalize the coordinates by dividing them by the image width and height
    yolo_center_x = center_x / image_width
    yolo_center_y = center_y / image_height

    # Normalize the width and height by dividing them by the image width and height
    yolo_width = width / image_width
    yolo_height = height / image_height

    # Return the bounding box in YOLO format
    return yolo_center_x, yolo_center_y, yolo_width, yolo_height

def segment_letters(image_path, ocr_predictions = None, train = True):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Unable to read image: {image_path}")
        return
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
    h, w, _ = image.shape
    annotations = []
    for bbox in letter_bboxes:
        x1, y1, x2, y2 = bbox
        bbox_yolo = convert_bbox_to_yolo((x1, y1, x2-x1, y2-y1), w, h)
        annotation = f"{bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}"
        annotations.append(annotation)

    if train:
        with open(os.path.join(ocr_predictions, image_path.split("\\")[1].split(".")[0]+".txt"), 'w') as f:
            f.write('\n'.join(annotations))
            f.write('\n')
    else:
        pass
    
    return letter_bboxes


#for i,filename in enumerate(os.listdir(test_images)):
#    f = os.path.join(test_images, filename)
#    segment_letters(f, ocr_predictions)
