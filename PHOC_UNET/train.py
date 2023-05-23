from tqdm.auto import tqdm
import wandb
from test import *

def train(model, train_loader, test_loader, criterion, optimizer, config, device = "cuda"):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        test2(model, test_loader, epoch, criterion, device)
        total_loss = 0
        for _, (images, labels) in enumerate(train_loader):

            loss = train_batch(images, labels, model, optimizer, criterion, device)
            total_loss += loss
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        train_log2(total_loss, len(train_loader.dataset), epoch)
        test2(model, test_loader, epoch, criterion, device)
    
    
    


def train_batch(images, labels, model, optimizer, criterion, device="cuda"):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss/example_ct}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss/example_ct:.3f}")

def train_log2(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "train loss": loss/example_ct}, step=epoch)
    print(f"Train Loss: {loss/example_ct:.3f}")