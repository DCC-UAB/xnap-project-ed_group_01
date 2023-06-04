

def convert_bbox_to_yolo(bbox, image_width, image_height):
    x, y, width, height = bbox

    # Calculate the bounding box center coordinates
    center_x = x + width / 2
    center_y = y + height / 2

    # Normalize the coordinates by dividing them by the image width and height
    yolo_center_x = center_x / image_width
    yolo_center_y = center_y / image_height

    # Normalize the width and height by dividing them by the image width and height
    yolo_width = width / image_width
    yolo_height = height / image_height

    # Return the bounding box in YOLO format
    return yolo_center_x, yolo_center_y, yolo_width, yolo_height