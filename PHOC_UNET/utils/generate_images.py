from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import sys
import string
import os

fonts = ["ARIAL.TTF", "CALIBRI.TTF", "COMIC.TTF"]

def generate_images(n, images_dir):
    for i in range(n):
        xy = [0,0]
        n = random.randint(3, 10)
        new_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))
        
        font = fonts[random.randint(0, len(fonts)-1)]
        font = ImageFont.truetype(f'C:/Users/xavid/Documents/GitHub/xnap-project-ed_group_01/generate_images/fonts/{font}', random.randint(25,40), encoding="unic")
        
        _, _, w, h = font.getbbox(new_str)
        offset = (random.randint(10, 20), random.randint(40,60))
        size = (w + offset[0], h + offset[1])
        img = Image.new("L", size=size, color= "white")

        draw = ImageDraw.Draw(img)
        xy = (offset[0]/2, offset[1]/2)
        draw.text(xy, new_str, font = font, fill = "black")

        img.save(os.path.join(images_dir, f"{i}_{new_str}.jpg"))


img_dir_train = "C:/Users/xavid/Desktop/Dataset_easy/train_easy"
img_dir_test = "C:/Users/xavid/Desktop/Dataset_easy/test_easy"
generate_images(10000, img_dir_train)
generate_images(1000, img_dir_test)