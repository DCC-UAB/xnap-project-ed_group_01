from tqdm import tqdm
import config
import wandb


def train_fn(model, data_loader, optimizer, epoch):
    model.train()
    fin_loss = 0
    example_ct = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for i, data in enumerate(tk0):
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
        example_ct += len(data[key])
        print(f"Loss after {str(epoch*len(data_loader.dataset) + example_ct).zfill(5)} examples: {loss.item():.3f}")
        wandb.log({"loss": loss.item()}, step=epoch*len(data_loader.dataset) + example_ct)
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)
        batch_preds, loss = model(**data)
        fin_loss += loss.item()
        fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader.dataset)
