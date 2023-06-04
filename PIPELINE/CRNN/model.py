import torch.nn as nn
#from torchinfo import summary


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same')
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same')
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding='same')
        self.relu5 = nn.ReLU()
        self.batch_norm5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same')
        self.relu6 = nn.ReLU()
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv7 = nn.Conv2d(512, 512, kernel_size=(2, 2))
        self.relu7 = nn.ReLU()

        self.blstm1 = nn.LSTM(512, 128, bidirectional=True, batch_first=True, dropout=0.2)
        self.blstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True, dropout=0.2)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.batch_norm5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.batch_norm6(x)
        x = self.pool6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        x = x.squeeze(2)
        x = x.permute(0,2,1)

        x, _ = self.blstm1(x)
        x, _ = self.blstm2(x)

        x = self.fc(x) # DESPRÉS D'AIXÒ LA SHAPE ÉS DE (BATCH_SIZE, 31, NUM_LABELS)
        x = x.permute(1, 0, 2) # FENT AIXÒ ÉS (31, BATCH_SIZE, NUM_LABELS)

        return x

# Create an instance of the CRNN model
#char_list = string.ascii_letters+string.digits
#num_classes = len(char_list) + 1

# Define your model
#model = CRNN(num_classes)

# Print the model summary
#summary(model, input_size=(1, 32, 128))
#summary(model, input_size=(16, 1, 32, 128))