import os
parent = os.getcwd()

# GENERATE IMAGES
multiclass = True
folder = "oneclass_randomchars"
n_train_images = 10
n_test_images = 5

train_labels = f"{parent}/Datasets/{folder}/labels/train"
train_images = f"{parent}/Datasets/{folder}/images/train"
test_labels = f"{parent}/Datasets/{folder}/labels/test"
test_images = f"{parent}/Datasets/{folder}/images/test"
path_fonts = f"{parent}/generate_images/fonts/"

# YOLO
data_yaml = "/home/alumne/ProjecteNN/xnap-project-ed_group_01/28_05/Detection/YOLO/data.yaml"

# ORC
ocr_predictions = "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/data/OCR_predictions"

# RECOGNITION (CNN)
saved_model_cnn = "/home/alumne/xnap-project-ed_group_01/28_05/Recognition/saved_model"

# PIPELINE (YOLO)
yolo_image_path = "/home/alumne/xnap-project-ed_group_01/28_05/data/images/test/"
yolo_model_path = '/home/alumne/xnap-project-ed_group_01/YOLO_recognition/checkpoints/last.pt'
gt_yolo = "/home/alumne/xnap-project-ed_group_01/28_05/data/labels/test"

# PIPELINE (OCR+CNN i YOLO+CNN)
type_model = "resnet_iit"
model_cnn_entrenat = f'/home/alumne/ProjecteNN/xnap-project-ed_group_01/28_05/Recognition/saved_model/{type_model}.pt'
model_yolo_entrenat = "/home/alumne/ProjecteNN/xnap-project-ed_group_01/28_05/Detection/YOLO/models_entrenats/best_iit.pt"

# PIPELINE (YOLO FOR RECOGNITION AND DETECTION)
yolo_entrenat_recog_detect = "/home/alumne/ProjecteNN/xnap-project-ed_group_01/28_05/PIPELINE/YOLO/models_entrenats/char_best.pt"