import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lrs
import time
import torch.nn.functional as F
import string
import wandb

import pandas as pd

from dataset2 import *

from model import *

wandb.init(project="CRNN", group="grup1", name= "CRNN loss curve")

def decode_predictions(text_batch_logits):

    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = decode_to_text(text_tokens)
        #text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new

def decode(self, logits):
        tokens = logits.softmax(2).argmax(2).squeeze(1)
        
        tokens = ''.join([self.idx2char[token] 
                        if token != 0  else '-' 
                        for token in tokens.numpy()])
        tokens = tokens.split('-')
        
        text = [char 
                for batch_token in tokens 
                for idx, char in enumerate(batch_token)
                if char != batch_token[idx-1] or len(batch_token) == 1]
        text = ''.join(text)
        
        return text

def decode_to_text(dig_lst):
    # decoding each digit into output word
    char_list = string.ascii_letters + string.digits
    txt = ""
    for index in dig_lst:
        try:
            txt += char_list[index]
        except IndexError:
            print("Invalid index:", index)
    
    return txt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
learning_rate = 0.1
num_epochs = 10

mat_data_train = "C:/Users/adars/OneDrive/Escritorio/ProjecteNN/IIIT5K/traindata.mat"
mat_data_test = "C:/Users/adars/OneDrive/Escritorio/ProjecteNN/IIIT5K/testdata.mat"
img_dir = "C:/Users/adars/OneDrive/Escritorio/ProjecteNN/IIIT5K/"
train_dataset = Dataset(mat_data_train, img_dir, "traindata")
test_dataset = Dataset(mat_data_test, img_dir, "testdata")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

char_list = string.ascii_letters+string.digits
num_classes = len(char_list) + 1
crnn = CRNN(num_classes)
crnn.to(device)
crnn.apply(weights_init)

criterion = nn.CTCLoss(blank = 0)
optimizer = optim.Adam(crnn.parameters(), lr=learning_rate)
scheduler = lrs.StepLR(optimizer, step_size=5, gamma=0.8)

epoch_losses = []
iteration_losses = []
for epoch in range(num_epochs):
    epoch_loss_list = []
    crnn.train()

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
              'start epoch %d/%d:' % (epoch+1,num_epochs),'learning_rate =',scheduler.get_lr()[0])
    
    for images,labels,orig_labels,label_length,input_length in train_dataloader:
        optimizer.zero_grad()
        text_batch_logits = crnn(images.to(device))
        text_batch_logps = F.log_softmax(text_batch_logits, 2)
        loss = criterion(text_batch_logps, labels, input_length, label_length)
        iteration_loss = loss.item()

        if np.isnan(iteration_loss) or np.isinf(iteration_loss):
            continue
        
        iteration_losses.append(iteration_loss)
        epoch_loss_list.append(iteration_loss)

        loss.backward()
        nn.utils.clip_grad_norm_(crnn.parameters(), 5)
        optimizer.step()

    epoch_loss = np.mean(epoch_loss_list)
    print("Epoch:{}    Loss:{}".format(epoch+1, epoch_loss))
    epoch_losses.append(epoch_loss)

    crnn.eval()
    with torch.no_grad():
        val_loss = []
        for images,labels,orig_labels,label_length,input_length in test_dataloader:
            text_batch_logits_val = crnn(images.to(device))
            text_batch_logps_val = F.log_softmax(text_batch_logits_val, 2)
            loss_val = criterion(text_batch_logps_val, labels, input_length, label_length)
            val_loss.append(loss_val.item())
        val_epoch_loss = np.mean(val_loss)
        print("Epoch:{}    Val Loss:{}".format(epoch+1, val_epoch_loss))

    wandb.log({"Train Loss": epoch_loss})
    wandb.log({"Validation Loss": val_epoch_loss})
    scheduler.step()
