import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import *
from dataset import *
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from sklearn.metrics import accuracy_score

num_epochs = 25
batch_size = 16
learning_rate = 0.001

# Set the device to 'cuda' if available, else use 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = Compose([
    Resize((64, 64)),  # Resize the image to a specific size
    ToTensor(),  # Convert the image to a tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

image_dir = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/data/images/train'
annotation_file = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/data/labels/train_predict/predict_final.txt'
dataset = CharacterDataset(annotation_file, image_dir, transforms)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

model = CharacterClassifier(num_classes=36) 
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training i tal
for epoch in range(num_epochs):
    loss = 0.0
    correct = 0
    total = 0
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

    # Validation loop
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_predictions.extend(predicted.tolist())
            val_labels.extend(labels.tolist())

    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy}")


torch.save(model.state_dict(), 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/YOLOv8/CNN/saved_model/model.pt')