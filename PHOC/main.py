import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import wandb

from dataset import *
from model import *

# Set random seed for reproducibility
torch.manual_seed(42)

num_classes = 36

# Define hyperparameters
learning_rate = 0.0001
batch_size = 8
num_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "C:/Users/xavid/Desktop/Dataset_easy/train_easy/"
test_dir = "C:/Users/xavid/Desktop/Dataset_easy/test_easy/"
train_set = PHOCDataset(train_dir)
test_set = PHOCDataset(test_dir)

file_list = sorted(os.listdir(train_dir))
labels = [os.path.splitext(filename)[0].split("_")[1] for filename in file_list]
label = [label.lower() for label in labels]
        
# Convert label to PHOC representation
phoc_representations = build_phoc(labels, 'abcdefghijklmnopqrstuvwxyz0123456789', [1])
suma = np.sum(phoc_representations, axis=0)
weights = phoc_representations.shape[0]/(suma+1e-6)
weights = torch.tensor(weights).to(device)
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


# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
#criterion = CosineLoss()
weight_decay = 0.00005
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
#plateau_scheduler = ReduceLROnPlateau(optimizer, verbose=True)
"""base_lr = 0.001  # Base learning rate
max_lr = 0.01  # Maximum learning rate
step_size_up = len(train_loader)  # Number of steps for learning rate to increase
step_size_down = 2 * step_size_up
scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down, mode='triangular')"""
scheduler = CosineAnnealingLR(optimizer, T_max=10)

wandb.init(project='PHOC', group='group1')

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        print(loss.item())
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        if ((i + 1) % 20) == 0:
                wandb.log({"Loss": loss.iten()})
    
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
    wandb.log({"Train loss": train_loss, "Test loss": test_loss})
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')