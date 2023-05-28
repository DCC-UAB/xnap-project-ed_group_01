import sys
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05")
from params import *
from PIL import Image, ImageDraw

image = Image.open("C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/data/images/test/weekending.jpg")

def read_txt_file(file_path):
    result = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            numbers = [float(num) for num in line.strip().split()]
            result.append(numbers)
    return result

# Aquí només ho estem fent per una imatge, la idea seria tenir-hi el directori amb predictions i labels correctes
# i que iteri per tots els elmeents dels directoris de 1 en 1 i calculi metriquis i dongui una preciosn, recall general per TOT
file_path_prediction = "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/data/OCR_predictions/weekending.txt"
file_path_labels = "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05/data/labels/test/weekending.txt"
image_predictions = read_txt_file(file_path_prediction)
image_labels = read_txt_file(file_path_labels)

a = image_labels[0][1:]
b = image_predictions[0]

def convert(a):
    new_a = a.copy()
    new_a[0] = a[0] - a[2]/2
    new_a[1] = a[1] - a[3]/2
    new_a[2] = a[0] + a[2]/2
    new_a[3] = a[1] + a[3]/2
    return new_a

def get_iou(a, b, epsilon=1e-5):

    a = convert(a)
    b = convert(b)

    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

