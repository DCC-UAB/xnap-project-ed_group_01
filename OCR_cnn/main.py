import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import *
from dataset import *
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from sklearn.metrics import accuracy_score, precision_score, recall_score
#import Levenshtein
from validate import *
import wandb

wandb.init(project="OCR_CNN", group="grup1", name="OCR_CNN_test")


num_epochs = 10
batch_size = 64
learning_rate = 0.01

# Set the device to 'cuda' if available, else use 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = Compose([
    Resize((64, 64)),  # Resize the image to a specific size
    ToTensor(),  # Convert the image to a tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

image_dir  = "C:/Users/adars/OneDrive/Escritorio/ProjecteNN/mnt/ramdisk/max/90kDICT32px/1/2"
annotation_file = "C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/OCR_cnn/annotation.txt"
dataset = CharacterDataset(annotation_file, image_dir, transforms)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

model = CharacterClassifier(num_classes=62) 
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training i tal
step = 0
log_frequency = 100
for epoch in range(num_epochs):
    loss = 0.0
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

        loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_predictions.extend(predicted.tolist())
        train_labels.extend(labels.tolist())

        step += 1
        """
        if step % log_frequency == 0:
            step_loss = loss.item() / (step*batch_size)
            step_accuracy = correct / total
            step_precision = precision_score(train_labels, train_predictions, average='weighted')
            step_recall = recall_score(train_labels, train_predictions, average='weighted')

            print(f"Step {step} - Step train Loss: {step_loss:.4f} - Step Accuracy: {step_accuracy:.2f}%")
            wandb.log({"Step Train Loss": step_loss, "Step Train Accuracy": step_accuracy,
                       "Step Train Precision": step_precision, "Step Train Recall": step_recall})
            
            step_val_predictions, step_val_labels, step_val_loss =  validate_func(model, val_dataloader, device, criterion)
            step_val_epoch_loss = step_val_loss / len(val_dataloader)
            step_val_accuracy = accuracy_score(step_val_labels, step_val_predictions)
            step_val_precision = precision_score(step_val_labels, step_val_predictions, average='weighted')
            step_val_recall = recall_score(step_val_labels, step_val_predictions, average='weighted')

            wandb.log({"Step Validation Loss": step_val_epoch_loss, "Step Validation Accuracy": step_val_accuracy,
                       "Step Validation Precision": step_val_precision, "Step Validation Recall": step_val_recall})
        """

    epoch_accuracy = correct / total
    train_precision = precision_score(train_labels, train_predictions, average='weighted')
    train_recall = recall_score(train_labels, train_predictions, average='weighted')

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

    val_predictions, val_labels, val_loss =  validate_func(model, val_dataloader, device, criterion)

    #val_epoch_loss = val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_precision = precision_score(val_labels, val_predictions, average='weighted')
    val_recall = recall_score(val_labels, val_predictions, average='weighted')
    print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f} - Accuracy: {val_accuracy:.2f}%")

    wandb.log({"Train Loss": loss, "Train Accuracy": epoch_accuracy, "Train Precision": train_precision, "Train Recall": train_recall})
    wandb.log({"Validation Loss": val_loss,"Validation Accuracy": val_accuracy, "Validation Precision": val_precision, "Validation Recall": val_recall})

torch.save(model.state_dict(), '/home/alumne/ProjecteNN/xnap-project-ed_group_01/OCR_cnn/saved_model/model.pt')