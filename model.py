import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel , 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel , 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel , channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class AttentionBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(AttentionBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = nn.Conv2d(1,6,2,1,0)
        self.relu = nn.ReLU()
        self.block = AttentionBlock(default_conv,6,1)
        self.conv3 = nn.Conv2d(6,1,1,1,0)
        self.fc1 = nn.Linear(31,16)
        self.fc2 = nn.Linear(16,6)
        self.fc3 = nn.Linear(6,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.block(x))
        x = self.relu(self.conv3(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return self.sigmoid(x)
