import torch.nn as nn
import torch.nn.functional as F
import torch


class cnn(nn.Module):

    def __init__(self):

        super(cnn, self).__init__()

        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = nn.BatchNorm1d(245)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.linear1 = nn.Linear(in_features=245, out_features=100)
        self.linear2 = nn.Linear(in_features=100, out_features=2)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, padding=1)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, padding=1)

        x = torch.flatten(x, start_dim=1)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)

        return x

