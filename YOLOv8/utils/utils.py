import torch
import torch.nn

import shutil

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

def save_ckp(state, checkpoint_dir, epoch, is_best = None, best_model_dir = None):
    f_path = checkpoint_dir + '/' + f'checkpoint{epoch}.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir / 'best_model.pt'
        shutil.copyfile(f_path, best_fpath)

# model, optimizer, start_epoch, scheduler = load_ckp(ckp_path, model, optimizer, scheduler)
def load_ckp(checkpoint_fpath, model, optimizer = None, scheduler = None):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, checkpoint['epoch'], scheduler