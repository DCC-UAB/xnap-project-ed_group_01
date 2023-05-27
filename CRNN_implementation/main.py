import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import *
from dataset import *

from tqdm import tqdm

# Set the device for training (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the paths to your train and validation data
train_image_dir = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/CRNN_implementation/dataset/img2/train'
train_label_file = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/CRNN_implementation/dataset/annot2/labels_finals_train.txt'
val_image_dir = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/CRNN_implementation/dataset/img2/test'
val_label_file = 'C:/Users/adars/github-classroom/DCC-UAB/xnap-project-ed_group_01/CRNN_implementation/dataset/annot2/labels_finals_test.txt'

# Set the input shape and number of classes
input_shape = (32, 128)  # Assuming input images of size 32x128
num_classes = 26  # Number of characters (excluding the blank label)

# Set the hyperparameters
batch_size = 16
num_epochs = 10
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create train and validation datasets
train_dataset = TextRecognitionDataset(train_image_dir, train_label_file, transform)
val_dataset = TextRecognitionDataset(val_image_dir, val_label_file, transform)

# Create train and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Create an instance of the CRNN model
model = CRNN(num_classes).to(device)

# Define the loss function (CTC loss) and optimizer
criterion = nn.CTCLoss(blank=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    # Iterate over the training batches
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        _, bs, _ = outputs.shape

        log_probs = F.log_softmax(outputs, 2)
        input_lengths = torch.full(
            size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
        )
        target_lengths = torch.zeros(bs, dtype=torch.int32)
        for i, t in enumerate(labels):
            target_lengths[i] = (sum([1 for t2 in t if t2 == 0]))
        loss = nn.CTCLoss(blank=0)(
            log_probs, labels, input_lengths, target_lengths
        )
        
        # Calculate the CTC loss
        #loss = criterion(outputs, labels, output_lengths, label_lengths)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.update(1)

    # Compute average training loss
    train_loss = running_loss / len(train_loader)

    # Evaluate on the validation set
    model.eval()  # Set the model to evaluation mode
    val_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Convert the output logits to predicted labels
            _, predicted = torch.max(outputs, dim=2)
            predicted = predicted.cpu().numpy()

            for prediction in predicted:
                text = ""
                for char in prediction:
                    if char != num_classes:  # Exclude the blank label
                        text += chr(char + ord('a'))  # Convert label to character
                val_predictions.append(text)

    # Print validation predictions
    print("Epoch:", epoch + 1)
    print("Validation Predictions:", val_predictions)

    # Print average training loss
    print("Train Loss:", train_loss)
    progress_bar.close()
