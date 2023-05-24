from tqdm.auto import tqdm
import torch
import wandb
from test import test, test2

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, config, device = "cuda"):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)
    model.train()
    # Run training and track with wandb
    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        total_loss = 0
        for _, (images, phoc_labels, text_labels) in enumerate(train_loader):

            loss = train_batch(images, phoc_labels, model, optimizer, criterion, device)
            total_loss += loss
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            #if ((batch_ct + 1) % 25) == 0:
            if epoch != 0:
                train_log(loss, example_ct, len(images), epoch, total_loss, test_loss)
        train_log2(total_loss, len(train_loader.dataset), example_ct, epoch)
        test_loss = test2(model, test_loader, epoch, criterion, device)
        scheduler.step()
        print(scheduler._last_lr)


def train_batch(images, labels, model, optimizer, criterion, device="cuda"):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs.float(), labels.float())
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, total_example_ct, example_ct, epoch, train_loss, test_loss):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss/example_ct}, step=total_example_ct)
    print(f"Loss after {str(total_example_ct).zfill(5)} examples: {loss/example_ct:.3f}")

def train_log2(loss, example_ct_batch ,example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "train-loss": loss/example_ct_batch}, step=example_ct)
    print(f"Train Loss: {loss/example_ct_batch:.3f}")

def lr_log(lr, epoch):
    wandb.log({"learning-rate": lr}, step=epoch)