import torch
import torch.nn as nn
import torch.nn.functional as F
from init import device, Args

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    
    def get_feature(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

class ApproxContainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Create neural networks
        self.model = ResNet9().to(device)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Args.lr, weight_decay=Args.weight_decay)
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=Args.lr, 
                                                             epochs=Args.total_epochs, 
                                                             steps_per_epoch=100000)  

    def predict(self, x):
        x = torch.Tensor(x).to(device)
        with torch.no_grad():
            y = self.model(x)
            pred = y.argmax(dim=1, keepdim=True)
            return pred
