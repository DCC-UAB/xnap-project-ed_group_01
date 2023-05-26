import wandb
from torchvision import transforms as T
from PIL import ImageDraw, ImageFont
from torchvision import transforms
from utils.predict_with_PHOC import predict_with_PHOC

def train_log(loss, total_example_ct):
    wandb.log({"loss": loss}, step=total_example_ct)
    print(f"Loss after {str(total_example_ct).zfill(5)} examples: {loss:.3f}")

def train_test_log(train_loss, test_loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "train loss": train_loss, "test loss": test_loss}, step=example_ct)
    print(f"Train Loss: {train_loss:.3f}\nTest Loss: {test_loss:.3f}")

def log_images(images, outputs, text_labels, epoch, mode):
    predicted_labels = predict_with_PHOC(outputs)
    t = transforms.Compose([transforms.Normalize(0, 1/0.1),transforms.Normalize(-0.5, 1)])
    t_images = t(images)
    images_with_labels = draw_images(t_images, text_labels, predicted_labels)
    wandb.log({f"Epoch{epoch}-{mode}": [wandb.Image(im) for im in images_with_labels]})

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
    font = ImageFont.truetype(f'generate_images/fonts/ARIAL.TTF', 10)
    draw.text((0,0), text, font=font, fill = color)
    return image

def lr_log(lr, epoch):
    wandb.log({"learning-rate": lr}, step=epoch)