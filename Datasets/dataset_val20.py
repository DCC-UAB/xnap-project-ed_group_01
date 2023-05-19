
path = '/Users/abriil/github-classroom/DCC-UAB/xnap-project-ed_group_01/Datasets/'

with open(path+'annotation_val.txt', "r") as file:
        im_paths = file.readlines()

im_paths = [p for p in im_paths if int(p.split('/')[1])<2697 and int(p.split('/')[1])>2676 ]

with open(path+'annotation_val20.txt', 'w') as f:
    f.write(''.join(im_paths))