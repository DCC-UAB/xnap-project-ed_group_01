import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import *
from dataset import *
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from sklearn.metrics import accuracy_score, precision_score, recall_score
import wandb
import sys
import os
sys.path.insert(0, "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/28_05")
from params import *

run_explication = "CNN test"
wandb.init(project="CNN", group="grup1", name= run_explication)

num_epochs = 25
batch_size = 64
learning_rate = 0.001
type_model = "resnet"

# Set the device to 'cuda' if available, else use 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = Compose([
    Resize((64, 64)),  # Resize the image to a specific size
    ToTensor(),  # Convert the image to a tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

dataset_train = CharacterDataset(train_labels, train_images, transforms)
dataset_test = CharacterDataset(test_labels, test_labels, transforms)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset_test, batch_size=batch_size)

model = CharacterClassifier(num_classes=36, type_model = type_model) 
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training i tal
for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    train_predictions = []
    train_labels = []
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_predictions.extend(predicted.tolist())
        train_labels.extend(labels.tolist())

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_precision = precision_score(train_labels, train_predictions, average='weighted')
    train_recall = recall_score(train_labels, train_predictions, average='weighted')
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

    # Validation loop
    model.eval()
    val_predictions = []
    val_labels = []
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_predictions.extend(predicted.tolist())
            val_labels.extend(labels.tolist())
            val_loss += criterion(outputs, labels).item()

    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_precision = precision_score(val_labels, val_predictions, average='weighted')
    val_recall = recall_score(val_labels, val_predictions, average='weighted')
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy}")

    wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_accuracy, "Train Precision": train_precision, "Train Recall": train_recall})
    wandb.log({"Validation Loss": val_loss,"Validation Accuracy": val_accuracy, "Validation Precision": val_precision, "Validation Recall": val_recall})


torch.save(model.state_dict(), os.path.join(saved_model_cnn, f'{type_model}.pt'))