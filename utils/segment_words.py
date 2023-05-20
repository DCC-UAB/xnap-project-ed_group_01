import cv2
import os
import pytesseract

def segment_letters(image_path):
    image = cv2.imread(image_path)
    img_original = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        thresh_inv = 255 - thresh
        contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        letter_bboxes.append((x, y, x+w, y+h))

    letter_bboxes.sort(key=lambda bbox: bbox[0])

    letters = []
    for bbox in letter_bboxes:
        x1, y1, x2, y2 = bbox
        letter_img = gray[y1:y2, x1:x2]
        letters.append(letter_img)

    return letter_bboxes, letters, img_original

def show_bounding_boxes(letter_bboxes, image):
    output_image = image.copy()

    for bbox in letter_bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Bounding Boxes', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


directory  = './mnt/ramdisk/max/90kDICT32px/1/1'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    bounding_boxes, segmented_letters, img = segment_letters(f)

    #print(bounding_boxes)

    show_bounding_boxes(bounding_boxes, img)

