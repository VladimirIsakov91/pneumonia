import torch.nn as nn
import torch.nn.functional as F
import torch


class Maxout(nn.Module):

    def __init__(self, in_neurons, out_neurons):

        super(Maxout, self).__init__()
        self.linear1 = nn.Linear(in_neurons, out_neurons)
        self.linear2 = nn.Linear(in_neurons, out_neurons)

    def forward(self, x):

        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x = torch.max(x1, x2)
        return x


class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = nn.BatchNorm1d(490)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=5, out_channels=2, kernel_size=(1, 1))
        #self.global_pool = nn.AdaptiveMaxPool3d((2, 1, 1))
        self.linear1 = nn.Linear(in_features=490, out_features=100)
        self.linear2 = nn.Linear(in_features=100, out_features=2)
        self.maxout1 = Maxout(245, 100)
        self.maxout2 = Maxout(100, 2)

    def forward(self, x):

        #x = self.dropout(x)

        x = self.conv1(x)
        x = F.leaky_relu_(x)
        x = F.max_pool2d(x, kernel_size=3, padding=1) + F.avg_pool2d(x, kernel_size=3, padding=1)

        x = self.conv2(x)
        x = F.leaky_relu_(x)
        x = F.max_pool2d(x, kernel_size=3, padding=1) + F.avg_pool2d(x, kernel_size=3, padding=1)

        #x = self.conv3(x)
        #x = F.relu(x)
        #x = F.max_pool2d(x, kernel_size=3, padding=1)

        #x = self.conv4(x)
        #x = torch.unsqueeze(x, 1)
        #x = self.global_pool(x)
        #x = torch.squeeze(x)

        x = torch.flatten(x, start_dim=1)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = F.leaky_relu_(x)

        x = self.linear2(x)

        return x

