import torch 

def validate_func(model, val_dataloader, device, criterion):
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
    return val_predictions, val_labels, val_loss