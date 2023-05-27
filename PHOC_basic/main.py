import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from dataset import *
from model import *

# Set random seed for reproducibility
torch.manual_seed(42)

num_classes = 36

# Define hyperparameters
learning_rate = 0.01
batch_size = 10
num_epochs = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


root_dir = '/content/drive/MyDrive/2'
dataset = PHOCDataset(root_dir)

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.1, random_state=42)

# Create data loaders for training and testing
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Create an instance of the PHOCNet model
model = PHOCNet(num_classes).to(device)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(weights_init)

import torch
import torch.nn as nn


class CosineLoss(nn.Module):

    def __init__(self, size_average=True, use_sigmoid=True):
        super(CosineLoss, self).__init__()
        self.averaging = size_average
        self.use_sigmoid = use_sigmoid

    def forward(self, input_var, target_var):
        '''
            Cosine loss:
            1.0 - (y.x / |y|*|x|)
        '''
        if self.use_sigmoid:
            loss_val = sum(1.0 - nn.functional.cosine_similarity(torch.sigmoid(input_var), target_var))
        else:
            loss_val = sum(1.0 - nn.functional.cosine_similarity(input_var, target_var))
        if self.averaging:
            loss_val = loss_val/input_var.data.shape[0]
        return loss_val

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
#criterion = CosineLoss()
weight_decay = 0.0005
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#plateau_scheduler = ReduceLROnPlateau(optimizer, verbose=True)
base_lr = 0.001  # Base learning rate
max_lr = 0.01  # Maximum learning rate
step_size_up = len(train_loader)  # Number of steps for learning rate to increase
step_size_down = 2 * step_size_up
scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down, mode='triangular')

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
    
    # Compute average training loss
    train_loss /= len(train_set)
    
    # Testing loop
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
    
    # Compute average test loss and accuracy
    test_loss /= len(test_set)
    #plateau_scheduler.step(test_loss)
    scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
