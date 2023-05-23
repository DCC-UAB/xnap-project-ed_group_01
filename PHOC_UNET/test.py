import wandb
import torch
from torchvision import transforms as T

def test(model, test_loader, device="cuda", save:bool= True):
    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
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


def test2(model, test_loader, epoch, criterion, device="cuda", save:bool= True):
    # Run the model on some test examples
    with torch.no_grad():
        total_loss = 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels)

            if i == 0:
                log_images(images, epoch, 5)
        
        test_log(total_loss, len(test_loader.dataset), epoch)

def test_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "test loss": loss/example_ct}, step=epoch)
    print(f"Test Loss: {loss/example_ct:.3f}")

def log_images(images, epoch, n):
    transform = T.ToPILImage()
    images_pil = [transform(im)for im in images[:n]]
    wandb.log({f"Test epoch {epoch}": [wandb.Image(images) for im in images_pil]})