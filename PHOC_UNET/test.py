import wandb
import torch

from utils.wandb_logs import *

def test(model, test_loader, device="cuda", save:bool= True):
    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, phoc_labels, text_labels in test_loader:
            images, labels = images.to(device), phoc_labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    if save:
        print(len(images))
        # Save the model in the exchangeable ONNX format
        torch.onnx.export(model,  # model being run
                          images,  # model input (or a tuple for multiple inputs)
                          "model.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        wandb.save("model.onnx")


def test2(model, test_loader, train_loader, epoch, criterion, device="cuda", save:bool= True):
    # Run the model on some test examples
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (images, phoc_labels, text_labels) in enumerate(test_loader):
            images, phoc_labels = images.to(device), phoc_labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs.float(), phoc_labels.float())
            if i == 0:
                log_images(images, outputs[:5].cpu().numpy(), text_labels[:5], epoch, "Test")
                
        for i, (images, phoc_labels, text_labels) in enumerate(train_loader):
            images, phoc_labels = images.to(device), phoc_labels.to(device)
            outputs = model(images)
            log_images(images, outputs[:5].cpu().numpy(), text_labels[:5], epoch, "Train")
            break
        
    return total_loss/len(test_loader)
