from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import sys
import os
sys.path.insert(0, '/home/alumne/ProjecteNN/xnap-project-ed_group_01')
from YOLOv8.utils.utils import convert_bbox_to_yolo


fonts = ["ARIAL.TTF", "CALIBRI.TTF", "COMIC.TTF", "CORBEL.TTF", "SEGOEPR.TTF"]
list_char = "123456789abcdefghijklmnopqrstuvwxyz"


def generate_images(n, label_dir, images_dir):
    for i in range(n):
        with open(os.path.join(label_dir, f"{i}.txt"), 'w') as file:
        #with open(f"./data/labels/{i}.txt", "w") as file:

            size = (random.randint(100, 150), random.randint(40,60))
            img = Image.new(mode="RGB", size=size, color='white')
            font = fonts[random.randint(0, len(fonts)-1)]
            font = ImageFont.truetype(f'/home/alumne/ProjecteNN/xnap-project-ed_group_01/generate_images/fonts/{font}', random.randint(15,30), encoding="unic")

            for j in range(random.randint(2, 6)):
                char_index = random.randint(0, len(list_char)-1)
                char = list_char[char_index]
                bbox = list(font.getbbox(char))
                
                
                width_box = bbox[2]-bbox[0]
                height_box = bbox[3]-bbox[1]
                x = random.randint(-bbox[0],size[0]-bbox[0]-width_box) 
                y = random.randint(-bbox[1],size[1]-bbox[1]-height_box)
                draw = ImageDraw.Draw(img)
                draw.text((x,y), char, font=font, fill = "black")

                bbox[0] =  max(0, bbox[0] +x)
                bbox[2] = min(size[0], bbox[2] +x)
                bbox[1] = max(0, bbox[1] +y)
                bbox[3] = min(size[1], bbox[3] +y)
                
                #draw.rectangle(bbox, None, "#f00")

                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]

                bbox_yolo = convert_bbox_to_yolo(bbox, size[0], size[1])
                char_index = 0 #comment for multiclass
                file.write(f"{char_index} {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")
            file.close()
            img.save(os.path.join(images_dir, f"{i}.jpg"))

train_labels = '/home/alumne/data/labels/train'
train_images = '/home/alumne/data/images/train'
test_labels = '/home/alumne/data/labels/test'
test_images = '/home/alumne/data/images/test'
generate_images(40000, train_labels, train_images)
generate_images(10000, test_labels, test_images)