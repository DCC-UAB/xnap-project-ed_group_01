import torch
import torch.nn 
from torchvision import transforms
from models.PHOCNET import *
from models.UNET import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataset import dataset


def get_data(annotation_file, img_dir, transform=None, slice=1, train=True):

    dataset_ = dataset(annotation_file, img_dir, train, transform)

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
        transforms.Resize((64, 64), antialias=True),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
    ])

    train, test = get_data(config.train_annotations, config.img_dir, transforms_train, train=True), get_data(config.test_annotations, config.img_dir, transforms_test, train=False)

    train_loader = make_loader(train, config.batch_size)
    test_loader = make_loader(test, config.batch_size)

    # Make the model
    model = PHOCNet(n_out = train[0][1].shape[0], input_channels = 3).to(device)
    #model = U_Net(in_ch= 3, out_ch = train[0][1].shape[0]).to(device)
    model.init_weights()

    # Make the loss and optimizer
    criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    return model, train_loader, test_loader, criterion, optimizer, scheduler
