import torch.nn as nn
import torch.nn.functional as F
import torch


class FCN(nn.Module):

    def __init__(self):

        super(FCN, self).__init__()

        self.dropout = nn.Dropout2d(p=0.2)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.batch_norm4 = nn.BatchNorm2d(16)
        self.batch_norm5 = nn.BatchNorm2d(2)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=(1, 1))

        self.global_pool = nn.AdaptiveMaxPool3d((2, 1, 1))
        self.padding = 1

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.leaky_relu_(x)
        x = F.max_pool2d(x, kernel_size=3, padding=self.padding) + F.avg_pool2d(x, kernel_size=3, padding=self.padding)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.leaky_relu_(x)
        x = F.max_pool2d(x, kernel_size=3, padding=self.padding) + F.avg_pool2d(x, kernel_size=3, padding=self.padding)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=3, padding=self.padding) + F.avg_pool2d(x, kernel_size=3, padding=self.padding)

        x = self.dropout(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=3, padding=self.padding) + F.avg_pool2d(x, kernel_size=3, padding=self.padding)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = F.leaky_relu(x)

        x = torch.unsqueeze(x, 1)
        x = self.global_pool(x)
        x = torch.squeeze(x)

        return x


