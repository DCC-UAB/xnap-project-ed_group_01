import wandb
import torch
from torchvision import transforms as T
from PIL import ImageDraw, ImageFont

from utils.predict_with_PHOC import predict_with_PHOC

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


def test2(model, test_loader, epoch, criterion, device="cuda", save:bool= True):
    # Run the model on some test examples
    with torch.no_grad():
        total_loss = 0
        for i, (images, phoc_labels, text_labels) in enumerate(test_loader):
            images, phoc_labels = images.to(device), phoc_labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, phoc_labels)

            if i == 0:
                predicted_labels = predict_with_PHOC(torch.sigmoid(outputs[:5]).cpu().numpy())
                images_with_labels = draw_images(images, text_labels[:5], predicted_labels)
                log_images(images_with_labels, epoch)
        
        test_log(total_loss, len(test_loader.dataset), epoch)

def test_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "test-loss": loss/example_ct}, step=epoch)
    print(f"Test Loss: {loss/example_ct:.3f}")

def log_images(images, epoch):
    wandb.log({f"Test epoch {epoch}": [wandb.Image(im) for im in images]})

def draw_images(images, text_labels, predicted_labels):
    transform = T.ToPILImage()
    images = [draw_one_image(transform(im), t_lab, p_lab) for im, t_lab, p_lab in zip(images, text_labels, predicted_labels)]
    return images

def draw_one_image(image, text_label, predicted_label):
    draw = ImageDraw.Draw(image)
    if text_label == predicted_label:
        color = "green"
    else:
        color = "red"
    text = text_label + "\n" + predicted_label
    font = ImageFont.truetype(f'generate_images/fonts/ARIAL.TTF', 5)
    draw.text((0,0), text, font=font, fill = color)
    return image
