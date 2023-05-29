train_labels = "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/prova/labels"
train_images = "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/prova/images"
test_labels = "/home/alumne/ProjecteNN/data_detection/IIT/labels/test"
test_images = "/home/alumne/ProjecteNN/data_detection/IIT/images/train"
path_fonts = "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/generate_images/fonts"

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