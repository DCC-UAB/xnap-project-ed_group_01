import os
import random
import wandb
import yaml

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train import *
from test import *
from utils.utils import *
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=cfg):
    # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config, device)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, config, device)
        
        # and test its final performance
        test(model, test_loader, device)

        return model

with open('params.yml', 'r') as file:
    configuration = yaml.safe_load(file)

if __name__ == "__main__":
    wandb.login()

    config = dict(
        train_annotations=configuration["train_annotations"],
        test_annotations=configuration["test_annotations"],
        img_dir= configuration["img_dir"],
        epochs=5,
        classes=10,
        kernels=[16, 32],
        batch_size=128,
        learning_rate=5e-3,
        architecture="CNN")
    model = model_pipeline(config)