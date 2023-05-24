from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import sys
import string
import os
sys.path.insert(0, 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01')
from YOLOv8.utils.utils import convert_bbox_to_yolo


fonts = ["ARIAL.TTF", "CALIBRI.TTF", "COMIC.TTF", "CORBEL.TTF", "SEGOEPR.TTF"]
list_char = "123456789abcdefghijklmnopqrstuvwxyz"


def generate_images(n, label_dir, images_dir, xy = (0,0)):
    for i in range(n):
        n = random.randint(3, 10)
        new_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))
        
        with open(os.path.join(label_dir, f"{new_str}.txt"), 'w') as file:
        #with open(f"./data/labels/{i}.txt", "w") as file:

            font = fonts[random.randint(0, len(fonts)-1)]
            font = ImageFont.truetype(f'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/generate_images/fonts/{font}', random.randint(15,30), encoding="unic")
            _, _, w, h = font.getbbox(new_str)
            size = (w + random.randint(2, 20), h + random.randint(2,20))
            img = Image.new(mode="RGB", size=size, color='white')

            draw = ImageDraw.Draw(img)
            draw.text(xy, new_str, font = font, fill = "black")
            for i, char in enumerate(new_str):
                _, _, right, _ = font.getbbox(new_str[:i+1])
                _, _, _, bottom = font.getbbox(char)
                width, height = font.getmask(char).size
                right += xy[0] # bbox[2]
                bottom += xy[1] # bbox[3]
                top = bottom - height #bbox[1]
                left = right - width #bbox[0]
                
                draw.rectangle((left, top, right, bottom), None, "#f00")

                bbox_yolo = convert_bbox_to_yolo((left, top, width, height), size[0], size[1])

                char_index = 0 #comment for multiclass
                file.write(f"{char_index} {bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]}\n")
            
            #char_index = random.randint(0, len(list_char)-1)
            #char = list_char[char_index]
            #bbox = list(font.getbbox(char))
            
            
            #width_box = bbox[2]-bbox[0]
            #height_box = bbox[3]-bbox[1]
            #x = random.randint(-bbox[0],size[0]-bbox[0]-width_box) 
            #y = random.randint(-bbox[1],size[1]-bbox[1]-height_box)
            #draw = ImageDraw.Draw(img)
            #draw.text((x,y), char, font=font, fill = "black")



            #bbox[0] =  max(0, bbox[0] +x)
            #bbox[2] = min(size[0], bbox[2] +x)
            #bbox[1] = max(0, bbox[1] +y)
            #bbox[3] = min(size[1], bbox[3] +y)
            
            #draw.rectangle(bbox, None, "#f00")

            #bbox[2] = bbox[2] - bbox[0]
            #bbox[3] = bbox[3] - bbox[1]

            #bbox_yolo = convert_bbox_to_yolo(bbox, size[0], size[1])
            file.close()
            img.save(os.path.join(images_dir, f"{new_str}.jpg"))


test_labels = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/test/test_annot'
test_images = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/test/test_img'
generate_images(5, test_labels, test_images)