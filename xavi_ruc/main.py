import random
import wandb
import yaml

import numpy as np
import torch

from train import train

from utils.utils import *
from tqdm.auto import tqdm

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(cfg:dict) -> None:
    with wandb.init(project="PHOCnet", config=cfg):
        config = wandb.config

        model, train_loader, test_loader, criterion, optimizer, scheduler = make(config, device)
        
        train(model, train_loader, test_loader, criterion, optimizer, scheduler, config, device)

        return model

with open('PHOC_UNET/params.yml', 'r') as file:
    configuration = yaml.safe_load(file)

if __name__ == "__main__":
    wandb.login()

    config = dict(
        train_dir=configuration["train_dir"],
        test_dir=configuration["test_dir"],
        epochs=50,
        batch_size= 16,
        learning_rate=0.01)
    model = model_pipeline(config)