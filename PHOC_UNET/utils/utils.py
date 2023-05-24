import torch
import torch.nn 
from torchvision import transforms
from models.PHOCNET import *
from models.UNET import *
from models.CNN_basic import *
from models.MLP_basic import *
from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR
from torchvision import datasets, models, transforms

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

def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

def make(config, device="cuda"):
    # Make the data
    
    transforms_train = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.Normalize(0.5, 0.1)
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.Normalize(0.5, 0.1)
    ])

    train, test = get_data(config.train_annotations, config.img_dir, transforms_train, train=True), get_data(config.test_annotations, config.img_dir, transforms_test, train=False)

    train_loader = make_loader(train, config.batch_size)
    test_loader = make_loader(test, config.batch_size)

    # Make the model
    #model = PHOCNet(n_out = train[0][1].shape[0], input_channels = 3).to(device)
    #model = U_Net(in_ch= 3, out_ch = train[0][1].shape[0]).to(device)
    #model = CNN_basic(n_out = train[0][1].shape[0]).to(device)
    #model = MLP_basic(n_out = train[0][1].shape[0]).to(device)
    #def init_weights(m):
    #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #        nn.init.kaiming_normal_(m.weight)
    #        if m.bias is not None:
    #            nn.init.constant_(m.bias, 0)
    #model.apply(init_weights)
    model = models.resnet18(pretrained=True) 
    set_parameter_requires_grad(model,True)
    model.fc = nn.Sequential(nn.Linear(512, 512),
                             nn.ReLU(),
                             nn.Linear(512, 512),
                             nn.ReLU(),
                             nn.Linear(512, train[0][1].shape[0]),
                             nn.Sigmoid())

    # Make the loss and optimizer
    criterion = nn.BCELoss(reduction = 'mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    #scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=1, step_size_up=4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    return model, train_loader, test_loader, criterion, optimizer, scheduler
