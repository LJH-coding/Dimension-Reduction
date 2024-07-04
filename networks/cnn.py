import torch
import torch.nn as nn

from init import device, Args

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        with torch.no_grad():
            tmp = torch.zeros(1, 3, 32, 32)
            n_flatten = self.cnn(tmp).shape[1]

        self.mlp = nn.Sequential(
            nn.Dropout(Args.dropout),
            nn.Linear(n_flatten, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(Args.dropout),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 10),
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x

    def get_feature(self, x):
        return self.cnn(x)

class ApproxContainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # create neural networks
        self.model = CNN().to(device)

        # create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Args.lr)

    def predict(self, x):
        x = torch.Tensor(x).to(device)
        with torch.no_grad():
            y = self.model(x)
            pred = y.argmax(dim=1, keepdim=True)
            return pred
