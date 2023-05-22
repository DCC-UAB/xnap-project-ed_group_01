import wandb
import torch
import torch.nn 
import torchvision
from torchvision import transforms
from models.PHOCNET import *
from models.UNET import *

import shutil
from PHOC.dataset import dataset


def get_data(annotation_file, img_dir, transform=None, slice=1, train=True):

    dataset_ = dataset(annotation_file, img_dir, transform)

    return dataset_

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader


def make(config, device="cuda"):
    # Make the data
    
    transforms_train = transforms.Compose([
        transforms.Resize((64, 64)),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),
    ])

    train, test = get_data(config.train_annotations, config.img_dir, transforms_train, train=True), get_data(config.test_annotations, config.img_dir, transforms_test, train=False)

    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    #model = PHOCNet(n_out = train[0][1].shape[0], input_channels = 3).to(device)
    model = U_Net(in_ch= 3, out_ch = train[0][1].shape[0]).to(device)
    model.init_weights()

    # Make the loss and optimizer
    criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer

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