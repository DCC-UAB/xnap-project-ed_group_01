import os
parent = os.getcwd()

# GENERATE IMAGES
type_generate = "random_chars" # Options: ["random_chars", "random_words"]
multiclass = True # Options: [True, False]
folder = "oneclass_randomchars"
n_train_images = 2
n_test_images = 1

train_labels = f"{parent}/Datasets/{folder}/labels/train"
train_images = f"{parent}/Datasets/{folder}/images/train"
test_labels = f"{parent}/Datasets/{folder}/labels/test"
test_images = f"{parent}/Datasets/{folder}/images/test"
path_fonts = f"{parent}/generate_images/fonts/"

# DETECTION (OCR)
ocr_predictions = f"{parent}/Datasets/ocr_predictions"

# DETECTION (YOLO)
data_yaml = f"{parent}/Detection/YOLO/data.yaml"
path_yolo = f"{parent}/Datasets/{folder}/"

# RECOGNITION (CNN)
saved_model_cnn = f"{parent}Recognition/saved_model"

# PIPELINE (YOLO)
yolo_image_path = test_images
yolo_model_path = '/home/alumne/xnap-project-ed_group_01/YOLO_recognition/checkpoints/last.pt'
gt_yolo = "/home/alumne/xnap-project-ed_group_01/28_05/data/labels/test"

# PIPELINE (OCR+CNN i YOLO+CNN)
type_model = "resnet_iit"
model_cnn_entrenat = f'/home/alumne/ProjecteNN/xnap-project-ed_group_01/28_05/Recognition/saved_model/{type_model}.pt'
model_yolo_entrenat = "/home/alumne/ProjecteNN/xnap-project-ed_group_01/28_05/Detection/YOLO/models_entrenats/best_iit.pt"

# PIPELINE (YOLO FOR RECOGNITION AND DETECTION)
yolo_entrenat_recog_detect = "/home/alumne/ProjecteNN/xnap-project-ed_group_01/28_05/PIPELINE/YOLO/models_entrenats/char_best.pt"