from tqdm.auto import tqdm
import wandb
from test import test, test2
from utils.wandb_logs import *

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, config, device = "cuda"):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)
    model.train()
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        train_loss = 0
        for _, (images, phoc_labels, _) in enumerate(train_loader):

            loss = train_batch(images, phoc_labels, model, optimizer, criterion, device)
            train_loss += loss.item()
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            #if ((batch_ct + 1) % 25) == 0:

            train_log(loss.item(), example_ct)
        test_loss = test2(model, test_loader, train_loader, epoch, criterion, device)
        
        train_test_log(train_loss/len(train_loader), test_loss, example_ct, epoch)
        
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