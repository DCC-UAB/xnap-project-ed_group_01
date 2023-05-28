from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import sys
import string
import os
from random_word import RandomWords

fonts = ["ARIAL.TTF", "CALIBRI.TTF", "COMIC.TTF"]

"""annotations_file = "Datasets/lexicon.txt"

with open(annotations_file, "r") as file:
    list_of_words = file.readlines()"""

r = RandomWords()

#list_of_words = [l[:-1] for l in list_of_words]

def generate_images(n, images_dir):
    for i in range(n):
        xy = [0,0]
        n = random.randint(3, 10)
        #new_str = list_of_words[random.randint(0, len(list_of_words)-1)]
        new_str = r.get_random_word()
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
generate_images(50, img_dir_train)
generate_images(10, img_dir_test)