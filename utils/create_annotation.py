import os
import random
import sys
import yaml
sys.path.insert(0, 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/')
with open('params.yml', 'r') as file:
    configuration = yaml.safe_load(file)

dir_imgs = configuration["img_dir"]
test_annot = "/home/alumne/Dataset/annotations/test_annotation.txt"
train_annot = "/home/alumne/Dataset/annotations/train_annotation.txt"

list_imgs = []

for path, subdirs, files in os.walk(dir_imgs):
   for filename in files:
    list_imgs.append(filename)

random.shuffle(list_imgs)

with open(train_annot, 'w') as file:
        iter_list = list_imgs[:int(len(list_imgs)*0.8)]
        for idx, row in enumerate(iter_list):
            if idx == len(iter_list)-1:
                file.write(row)
            else:
                 file.write(row+'\n')


with open(test_annot, 'w') as file:
        iter_list = list_imgs[int(len(list_imgs)*0.8):]
        for idx, row in enumerate(iter_list):
            if idx == len(iter_list)-1:
                file.write(row)
            else:
                 file.write(row+'\n')